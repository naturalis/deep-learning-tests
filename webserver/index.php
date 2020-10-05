<?php

    include_once("class.baseClass.php");
    include_once("class.htmlClass.php");

    $base = new BaseClass;
    $base->setProjectRoot(getenv('PROJECT_ROOT'));
    $base->setProjectName(getenv('PROJECT_NAME'));
    $base->setImagesRoot(getenv('IMAGES_ROOT'));

    $html = new HtmlClass;

    echo $html->header();

    echo $html->h1($base->getProjectName());
    echo $html->h2($base->getProjectRoot() . "(" . $base->getImagesRoot() . ")" );

    print_r($base->getModels());
