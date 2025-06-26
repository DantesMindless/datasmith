#!/bin/bash
redis-cli -a redis << EOF
AUTH redis redis
HSET product:1 name "Laptop" price 999.99 stock 50 category "Electronics"
HSET product:2 name "Phone" price 599.99 stock 100 category "Electronics"
HSET product:3 name "Desk" price 299.99 stock 25 category "Furniture"
HSET product:4 name "Chair" price 149.99 stock 75 category "Furniture"
HSET product:5 name "Monitor" price 399.99 stock 30 category "Electronics"

SET user:1 '{"id": 1, "name": "John", "email": "john@test.com"}'
SET user:2 '{"id": 2, "name": "Alice", "email": "alice@test.com"}'
SET user:3 '{"id": 3, "name": "Bob", "email": "bob@test.com"}'
SET user:4 '{"id": 4, "name": "Emma", "email": "emma@test.com"}'
SET user:5 '{"id": 5, "name": "Mike", "email": "mike@test.com"}'

LPUSH orders:user:1 "order:1001" "order:1002"
LPUSH orders:user:2 "order:2001" "order:2002" "order:2003"
LPUSH orders:user:3 "order:3001"

SADD categories "Electronics" "Furniture" "Books" "Clothing" "Sports"

ZADD product:ratings 4.5 "product:1"
ZADD product:ratings 4.2 "product:2"
ZADD product:ratings 4.8 "product:3"
ZADD product:ratings 4.1 "product:4"
ZADD product:ratings 4.6 "product:5"
EOF
