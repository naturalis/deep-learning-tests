<?php

    class BaseClass
    {

        private $projectRoot;
        private $projectName;
        private $imagesRoot;
        private $modelList;
        private $models;
        private $model;
        private $classes;
        private $projectWebRoot = "/project";

        private $datasetFile = "dataset.json";
        private $analysisFile = "analysis.json";
        private $classesFile = "classes.json";
        private $classesListFile = "classes.csv";
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

        public function getProjectWebRoot()
        {
            return $this->projectWebRoot;
        }

        public function getProjectName()
        {
            return $this->projectName;
        }

        public function getImagesRoot()
        {
            return $this->imagesRoot;
        }

        public function setModel($model)
        {
            $this->model = $model;
        }

        public function getModel()
        {
            return $this->model;
        }

        public function getDataset()
        {
            return json_decode(file_get_contents(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->datasetFile])),true);
            
        }

        public function getAnalysis()
        {
            try
            {
                return json_decode(file_get_contents(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->analysisFile])),true);
            }
            catch (Exception $e)
            {
                return [ "error" : $e->getMessage() ];
            }
        }

        public function getClasses()
        {
            if (($handle = fopen(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->classesListFile]), "r")) !== FALSE)
            {
                while (($row = fgetcsv($handle, 1000, ",")) !== FALSE)
                {
                     $this->classes[] = [ "support" => $row[1], "name" => $row[0] ];
                }

                fclose($handle);
            }

            $c = json_decode(file_get_contents(implode("/",[$this->getProjectRoot(),"models",$this->model,$this->classesFile])),true);

            foreach ($c as $class => $val)
            {
                $key = array_search($class, array_column($this->classes, "name"));
                $this->classes[$key]["key"] = $val;
            }

            return $this->classes;
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
                $analysis = $this->getAnalysis();
                $this->model = $model;
                $this->models[] = [
                    "model" => $model,
                    "size" => $this->getModelSize(),
                    "dataset" => $this->getDataset(),
                    "analysis" => !isset($analysis["error"]) ? $analysis : null,
                    "accuracy" => !isset($analysis["error"]) ? $analysis["classification_report"]["accuracy"] : null,
                ];
            }
        }

        public function getModels($sort="model",$order="asc")
        {
            usort(
                $this->models,
                function($a,$b) use ($sort,$order)
                {
                    return  $a[$sort] == $b[$sort] ? 0 : ($order=="asc" ? $a[$sort] > $b[$sort] : $a[$sort] < $b[$sort]);
                });

            return $this->models;
        }

        public function getModelImagePath($image)
        {
            return implode("/",[$this->getProjectWebRoot(),"models",$this->model,$image]);
        }

    }