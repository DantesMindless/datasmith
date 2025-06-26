class DatabaseRouter:
    """
    A router to control database operations for models whose table names start with 'test_'.
    """

    def db_for_read(self, model, **hints):
        """
        Direct read operations for models whose table names start with 'test_' to the 'test' database.
        """
        if model._meta.db_table.startswith("test_"):
            return "test"
        return "default"

    def db_for_write(self, model, **hints):
        """
        Direct write operations for models whose table names start with 'test_' to the 'test' database.
        """
        if model._meta.db_table.startswith("test_"):
            return "test"
        return "default"

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if models are in the same database.
        """
        db_set = {"default", "test"}
        if obj1._state.db in db_set and obj2._state.db in db_set:
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        print(f"allow_migrate: db={db}, app_label={app_label}, model_name={model_name}")

        # Use hints to determine the model's db_table
        model = hints.get("model")
        if model:
            table_name = model._meta.db_table
            # Allow migrations for models whose table names start with 'test_'
            if table_name.startswith("test_"):
                allowed = db == "test"
                print(f"allow_migrate decision for {table_name} in db={db}: {allowed}")
                return allowed

        # Default behavior for models not starting with 'test_'
        default_allowed = db == "default"
        print(
            f"allow_migrate decision for model_name={model_name} (default): {default_allowed}"
        )
        return default_allowed
