<?php

    include_once("class.baseClass.php");
    include_once("class.htmlClass.php");

    $base = new BaseClass;
    $base->setProjectRoot(getenv('PROJECT_ROOT'));
    $base->setProjectName(getenv('PROJECT_NAME'));
    $base->setImagesRoot(getenv('IMAGES_ROOT'));
    $base->setModels();

    $html = new HtmlClass;

    echo $html->header();

    echo $html->h1($base->getProjectName());
    echo $html->h2("project root: " . $base->getProjectRoot());
    echo $html->h2("image root: " . $base->getImagesRoot());
    echo $html->p("available models:");

    $l=[];
    foreach ($base->getModels() as $model)
    {
        $l[]=
            vsprintf("%s %s %s %s %s %s %s %s",
                [
                    $model["model"],
                    // $model["dataset"]["state"]
                    $model["dataset"]["created"],
                    $model["dataset"]["model_note"],
                    $model["dataset"]["class_count"],
                    $model["dataset"]["class_count_before_maximum"],
                    $model["dataset"]["class_image_minimum"],
                    $model["dataset"]["class_image_maximum"],
                    $model["analysis"]["classification_report"]["accuracy"],
                ]
            );

    }

    echo $html->list($l);
