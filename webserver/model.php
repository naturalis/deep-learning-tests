<?php

    include_once("class.baseClass.php");
    include_once("class.htmlClass.php");

    $base = new BaseClass;
    $base->setProjectRoot(getenv('PROJECT_ROOT'));
    $base->setProjectName(getenv('PROJECT_NAME'));
    $base->setImagesRoot(getenv('IMAGES_ROOT'));
    $base->setModel($_GET["id"]);

    $html = new HtmlClass;

    echo $html->header();

    echo $html->h1($base->getProjectName());
    echo $html->h2("project root: " . $base->getProjectRoot());
    echo $html->h2("model: " . $base->getModel());

    $size = $base->getModelSize();
    $dataset = $base->getDataset();
    $analysis = $base->getAnalysis();
    $classes = $base->getClasses();

    $m = $analysis["confusion_matrix"];

    $t=[];

    $t["a"][] = "";
    foreach ($m as $cKey => $col)
    {
        $key = array_search($cKey, array_column($classes, "key"));
        $t["a"][] = $classes[$key]["name"];
    }

    foreach ($m as $cKey => $col)
    {
        $key = array_search($cKey, array_column($classes, "key"));
        $t[$cKey][] = $classes[$key]["name"];
        foreach ($col as $rKey => $row)
        {
            $t[$cKey][] = $m[$rKey][$cKey];
        }  
    }

    echo $html->table($t,"confusion_matrix");

    echo "<pre>";
    print_r($classes);
    // print_r($analysis);
    // print_r($dataset);

?>
<script type="text/javascript">
$( document ).ready(function()
{
    $("#confusion_matrix tr td").mouseover(function()
    {
        console.log($(this).html());
    });
});
</script>
