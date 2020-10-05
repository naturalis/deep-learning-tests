<?php

    class BaseClass
    {

        private $projectRoot;
        private $projectName;
        private $imagesRoot;
        private $modelList;
        private $models;
        private $model;

        private $datasetFile = "dataset.json";
        private $analysisFile = "analysis.json";
        private $modelFile = "model.hdf5";

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
            $this->imagesRoot = $var;
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
            return $this->imagesRoot;
        }

        public function getDataset()
        {
            return json_decode(file_get_contents(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->datasetFile])),true);
            
        }
        public function getAnalysis()
        {
            return json_decode(file_get_contents(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->analysisFile])),true);
        }

        public function getModelSize()
        {
            return filesize(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->modelFile]));
        }

        public function setModels()
        {
            $this->modelList = array_filter(
                scandir(implode("/",[$this->getProjectRoot(),"models"])),
                function($a){ return !in_array($a,[".",".."]);}
            );

            foreach ($this->modelList as $key => $model)
            {
                $this->model = $model;
                $this->models[] = [
                    "model" => $model,
                    "size" => $this->getModelSize(),
                    "dataset" => $this->getDataset(),
                    "analysis" => $this->getAnalysis(),
                    "accuracy" => $this->getAnalysis()["classification_report"]["accuracy"]
                ];
            }
        }

        public function getModels($sort="model",$order="asc")
        {
            usort(
                $this->models,
                function($a,$b) use ($sort,$order)
                {


                    return
                        $a[$sort] == ($b[$sort] ? 0 : ($a[$sort] > $b[$sort]));


                });
            return $this->models;
        }

    }