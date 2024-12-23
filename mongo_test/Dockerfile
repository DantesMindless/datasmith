FROM mongo:latest

COPY init/ /docker-entrypoint-initdb.d/
RUN chmod 777 /docker-entrypoint-initdb.d/*.js
