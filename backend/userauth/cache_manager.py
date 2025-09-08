"""
Advanced caching manager for permission system with Redis support
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from django.core.cache import cache
from django.conf import settings
from django.core.cache.backends.redis import RedisCache
from .logging_utils import permission_logger

logger = logging.getLogger(__name__)


class PermissionCacheManager:
    """Advanced caching manager for permission operations"""
    
    # Cache key patterns
    USER_DATASOURCE_ACCESS = "perm:ds_access:{user_id}:{datasource_id}"
    USER_COLUMNS = "perm:columns:{user_id}:{datasource_id}:{table_name}"
    USER_PERMISSION_SUMMARY = "perm:summary:{user_id}"
    USER_ROLE_LEVEL = "perm:role_level:{user_id}"
    DATASOURCE_USERS = "perm:ds_users:{datasource_id}"
    PERMISSION_STATS = "perm:stats:system"
    
    # Cache TTL configurations (in seconds)
    DEFAULT_TTL = getattr(settings, 'PERMISSION_CACHE_TTL', 300)  # 5 minutes
    SHORT_TTL = 60  # 1 minute for frequently changing data
    LONG_TTL = 1800  # 30 minutes for stable data
    
    def __init__(self):
        self.is_redis = isinstance(cache, RedisCache)
        self._pipeline_operations = []
        
    def _get_cache_key(self, pattern: str, **kwargs) -> str:
        """Generate cache key from pattern and parameters"""
        return pattern.format(**kwargs)
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize complex values for caching"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        return str(value)
    
    def _deserialize_value(self, value: str, value_type: type = None) -> Any:
        """Deserialize cached values"""
        if value is None:
            return None
        
        if value_type == dict or value_type == list:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        if value_type == bool:
            return value.lower() == 'true'
        
        if value_type == int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0
        
        return value
    
    def get_user_datasource_access(self, user_id: str, datasource_id: str) -> Optional[bool]:
        """Get cached datasource access result"""
        cache_key = self._get_cache_key(
            self.USER_DATASOURCE_ACCESS,
            user_id=user_id,
            datasource_id=datasource_id
        )
        result = cache.get(cache_key)
        return self._deserialize_value(result, bool) if result is not None else None
    
    def set_user_datasource_access(
        self, 
        user_id: str, 
        datasource_id: str, 
        has_access: bool,
        ttl: int = None
    ):
        """Cache datasource access result"""
        cache_key = self._get_cache_key(
            self.USER_DATASOURCE_ACCESS,
            user_id=user_id,
            datasource_id=datasource_id
        )
        cache.set(cache_key, str(has_access).lower(), ttl or self.DEFAULT_TTL)
    
    def get_user_columns(
        self, 
        user_id: str, 
        datasource_id: str, 
        table_name: str
    ) -> Optional[List[str]]:
        """Get cached accessible columns for user"""
        cache_key = self._get_cache_key(
            self.USER_COLUMNS,
            user_id=user_id,
            datasource_id=datasource_id,
            table_name=table_name
        )
        result = cache.get(cache_key)
        return self._deserialize_value(result, list) if result is not None else None
    
    def set_user_columns(
        self,
        user_id: str,
        datasource_id: str,
        table_name: str,
        columns: List[str],
        ttl: int = None
    ):
        """Cache accessible columns for user"""
        cache_key = self._get_cache_key(
            self.USER_COLUMNS,
            user_id=user_id,
            datasource_id=datasource_id,
            table_name=table_name
        )
        cache.set(cache_key, self._serialize_value(columns), ttl or self.DEFAULT_TTL)
    
    def get_user_permission_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user permission summary"""
        cache_key = self._get_cache_key(self.USER_PERMISSION_SUMMARY, user_id=user_id)
        result = cache.get(cache_key)
        return self._deserialize_value(result, dict) if result is not None else None
    
    def set_user_permission_summary(
        self, 
        user_id: str, 
        summary: Dict[str, Any],
        ttl: int = None
    ):
        """Cache user permission summary"""
        cache_key = self._get_cache_key(self.USER_PERMISSION_SUMMARY, user_id=user_id)
        cache.set(cache_key, self._serialize_value(summary), ttl or self.SHORT_TTL)
    
    def invalidate_user_cache(
        self, 
        user_id: str, 
        datasource_id: str = None, 
        table_name: str = None
    ):
        """Smart cache invalidation for user permissions"""
        keys_to_delete = []
        
        if datasource_id and table_name:
            # Invalidate specific table column cache
            keys_to_delete.append(
                self._get_cache_key(
                    self.USER_COLUMNS,
                    user_id=user_id,
                    datasource_id=datasource_id,
                    table_name=table_name
                )
            )
        elif datasource_id:
            # Invalidate datasource access and related column caches
            keys_to_delete.append(
                self._get_cache_key(
                    self.USER_DATASOURCE_ACCESS,
                    user_id=user_id,
                    datasource_id=datasource_id
                )
            )
            
            # Find column cache keys for this datasource
            if self.is_redis:
                pattern = self._get_cache_key(
                    self.USER_COLUMNS,
                    user_id=user_id,
                    datasource_id=datasource_id,
                    table_name="*"
                )
                keys_to_delete.extend(self._get_keys_by_pattern(pattern))
        else:
            # Invalidate all permission caches for user
            patterns = [
                self._get_cache_key(self.USER_DATASOURCE_ACCESS, user_id=user_id, datasource_id="*"),
                self._get_cache_key(self.USER_COLUMNS, user_id=user_id, datasource_id="*", table_name="*"),
                self._get_cache_key(self.USER_PERMISSION_SUMMARY, user_id=user_id),
                self._get_cache_key(self.USER_ROLE_LEVEL, user_id=user_id)
            ]
            
            for pattern in patterns:
                if self.is_redis and "*" in pattern:
                    keys_to_delete.extend(self._get_keys_by_pattern(pattern))
                else:
                    keys_to_delete.append(pattern)
        
        # Delete cache keys
        if keys_to_delete:
            cache.delete_many([key for key in keys_to_delete if "*" not in key])
            
            logger.info(
                f"Invalidated {len(keys_to_delete)} permission cache keys for user {user_id}",
                extra={
                    'user_id': user_id,
                    'datasource_id': datasource_id,
                    'table_name': table_name,
                    'cache_keys_invalidated': len(keys_to_delete)
                }
            )
    
    def _get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get cache keys matching pattern (Redis only)"""
        if not self.is_redis:
            return []
        
        try:
            # Use Redis SCAN for pattern matching
            redis_client = cache._cache.get_client()
            return list(redis_client.scan_iter(match=pattern))
        except Exception as e:
            logger.warning(f"Failed to get keys by pattern {pattern}: {e}")
            return []
    
    def warm_user_cache(self, user_id: str, datasources: List[str] = None):
        """Pre-warm cache for user's frequently accessed permissions"""
        from .models import DataSourcePermission, ColumnPermission
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return
        
        # Get user's datasource permissions
        ds_permissions = DataSourcePermission.objects.filter(user=user)
        if datasources:
            ds_permissions = ds_permissions.filter(datasource_id__in=datasources)
        
        # Cache datasource access
        for perm in ds_permissions:
            self.set_user_datasource_access(
                user_id=user_id,
                datasource_id=str(perm.datasource_id),
                has_access=True,
                ttl=self.LONG_TTL
            )
        
        # Get user's column permissions and group by datasource/table
        column_permissions = ColumnPermission.objects.filter(user=user)
        if datasources:
            column_permissions = column_permissions.filter(datasource_id__in=datasources)
        
        # Group columns by datasource and table
        columns_by_table = {}
        for perm in column_permissions:
            key = (str(perm.datasource_id), perm.table_name)
            if key not in columns_by_table:
                columns_by_table[key] = []
            columns_by_table[key].append(perm.column_name)
        
        # Cache column access
        for (datasource_id, table_name), columns in columns_by_table.items():
            self.set_user_columns(
                user_id=user_id,
                datasource_id=datasource_id,
                table_name=table_name,
                columns=columns,
                ttl=self.LONG_TTL
            )
        
        logger.info(
            f"Warmed cache for user {user_id}: {len(ds_permissions)} datasources, "
            f"{len(columns_by_table)} table-column combinations",
            extra={
                'user_id': user_id,
                'datasources_cached': len(ds_permissions),
                'table_combinations_cached': len(columns_by_table)
            }
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        stats = {
            'cache_backend': type(cache).__name__,
            'is_redis': self.is_redis,
            'estimated_permission_keys': 0,
            'cache_hit_info': 'Not available for this backend'
        }
        
        if self.is_redis:
            try:
                redis_client = cache._cache.get_client()
                
                # Count permission-related keys
                patterns = ['perm:*']
                total_keys = 0
                for pattern in patterns:
                    keys = list(redis_client.scan_iter(match=pattern, count=1000))
                    total_keys += len(keys)
                
                stats['estimated_permission_keys'] = total_keys
                
                # Get Redis info if available
                info = redis_client.info()
                stats['redis_info'] = {
                    'used_memory_human': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
                
                # Calculate hit rate
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                if hits + misses > 0:
                    hit_rate = hits / (hits + misses) * 100
                    stats['cache_hit_rate'] = f"{hit_rate:.2f}%"
                
            except Exception as e:
                logger.warning(f"Failed to get Redis cache stats: {e}")
                stats['error'] = str(e)
        
        return stats
    
    def clear_all_permission_cache(self):
        """Clear all permission-related cache entries"""
        if self.is_redis:
            try:
                redis_client = cache._cache.get_client()
                keys = list(redis_client.scan_iter(match='perm:*'))
                if keys:
                    redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} permission cache keys")
                    return len(keys)
            except Exception as e:
                logger.error(f"Failed to clear permission cache: {e}")
                return 0
        else:
            logger.warning("Bulk cache clearing only supported with Redis backend")
            return 0


# Global cache manager instance
cache_manager = PermissionCacheManager()