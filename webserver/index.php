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
    foreach ($base->getModels("accuracy","desc") as $model)
    {
        $l[]=
            vsprintf("%s (%s); accuracy: %s; %s classes (of %s; %s...%s images p/class); (%s) [%s]",
                [
                    $model["model"],
                    substr($model["dataset"]["created"],0,19),
                    $model["accuracy"],
                    $model["dataset"]["class_count"],
                    $model["dataset"]["class_count_before_maximum"],
                    $model["dataset"]["class_image_minimum"],
                    $model["dataset"]["class_image_maximum"],
                    $model["dataset"]["model_note"],
                    $model["dataset"]["state"]
                ]
            );

    }

    echo $html->list($l);
