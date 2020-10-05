<?php

    class BaseClass
    {

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
            return array_filter(scandir(implode("/",[$this->getProjectRoot(),"models"])),function($a){ return !in_array($a,[".",".."]);});
        }

    }