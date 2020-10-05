<?php

    class BaseClass {

        private $projectRoot;
        private $projectName;
        private $ImagesRoot;

        public function __init()
        {
        }

        public function setProjectRoot($var)
        {
            $this->projectRoot = rtrim($var,"/");
        }

        public function setProjectName($var)
        {
            $this->projectName = $var;
        }

        public function setImagesRoot($var)
        {
            $this->ImagesRoot = $var;
        }

        public function getProjectRoot()
        {
            return $this->projectRoot;
        }

        public function getProjectName()
        {
            return $this->projectName;
        }

        public function getImagesRoot()
        {
            return $this->ImagesRoot;
        }

        public function getModels()
        {
            return scandir(implode("/",$this->getProjectRoot(),"models"));
        }


    }



    $base = new BaseClass;
    $base->setProjectRoot(getenv('PROJECT_ROOT'));
    $base->setProjectName(getenv('PROJECT_NAME'));
    $base->setImagesRoot(getenv('IMAGES_ROOT'));

    echo $base->getProjectName() . "<br />";
    echo $base->getProjectRoot() . "<br />";
    echo $base->getImagesRoot() . "<br />";

    print_r($base->getModels());
