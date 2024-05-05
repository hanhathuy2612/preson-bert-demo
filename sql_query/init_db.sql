create table article
(
    id          int auto_increment primary key,
    title       varchar(255),
    description text(1000)
);

create table tag
(
    id   int auto_increment primary key,
    name varchar(255)
);

create table article_tag
(
    id         int auto_increment primary key,
    article_id int,
    tag_id     int,
    foreign key (article_id) references article (id),
    foreign key (tag_id) references tag (id)
);