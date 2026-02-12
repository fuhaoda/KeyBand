FROM nginx:alpine

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY web/ /usr/share/nginx/html/

