FROM nginx:alpine

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docs/ /usr/share/nginx/html/

