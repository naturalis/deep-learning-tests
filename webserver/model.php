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


    // classes

    $c=[];

    $c[$key][] = [ "html" => "class" ];
    $c[$key][] = [ "html" => "support" ];
    $c[$key][] = [ "html" => "f1-score" ];
    $c[$key][] = [ "html" => "precision" ];
    $c[$key][] = [ "html" => "recall" ];
    $c[$key][] = [ "html" => "top_3" ];
    $c[$key][] = [ "html" => "top_5" ];

    foreach ($classes as $cKey => $class)
    {
        if (!$class["key"])
        {
            continue;
        }

        $key = array_search($cKey, array_column($analysis["top_k_per_class"], "class"));

        $c[$key][] = [ "html" => $class["name"] ];
        $c[$key][] = [ "html" => $class["support"] ];
        // $c[$key][] = [ "html" => $analysis["classification_report"][$class["key"]]["support"] ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["f1-score"],2) ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["precision"],2) ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["recall"],2) ];
        // $c[$key][] = [ "html" => round($analysis["top_k_per_class"][$class["key"]]["top_1"] / $class["support"],2) ];
        $c[$key][] = [ "html" => round($analysis["top_k_per_class"][$class["key"]]["top_3"] / $class["support"],2) ];
        $c[$key][] = [ "html" => round($analysis["top_k_per_class"][$class["key"]]["top_5"] / $class["support"],2) ];
    }


    echo $html->p($html->table($c,"classes"));









    // confusion matrix

    $m = $analysis["confusion_matrix"];

    $t=[];

    $t["h"]["h"] = [ "html" => "" ];
    foreach ($m as $cKey => $col)
    {
        $key = array_search($cKey, array_column($classes, "key"));
        $t["h"][] = [ 
            "html" => $classes[$key]["key"],
            "title" => $classes[$key]["name"] 
        ];
    }

    foreach ($m as $cKey => $col)
    {
        $key = array_search($cKey, array_column($classes, "key"));
        $t[$cKey]["h"] = [
            "html" => $classes[$key]["key"] . ". " . $classes[$key]["name"],
            "title" => $classes[$key]["name"]
        ];

        foreach ($col as $rKey => $row)
        {
            $t[$cKey][] = [ "html" => $m[$rKey][$cKey] ];
        }  
    }

    echo $html->p($html->table($t,"confusion_matrix"));

    echo "<pre>";
    print_r($classes);
    print_r($analysis);
    print_r($dataset);

?>
<script type="text/javascript">
$( document ).ready(function()
{
    $("#confusion_matrix tr td").mouseover(function()
    {

        var r = $(this).attr('data-row');
        var c = $(this).attr('data-col');
        var a = $('td[data-col="'+c+'"][data-row="h"]').attr("title");
        var b = $('td[data-row="'+r+'"][data-col="h"]').attr("title");

        var sum = 0;
        $('td[data-col="'+c+'"][data-row!="h"]').each(function()
        {
            sum += parseInt($(this).html());
        })

        if (r!="h" && c!="h")
        {
            $(this).attr("title",$(this).html() + " (of "+ sum +")\n" + a + "\nidentified as\n" + b);
            // console.log(a + "\\" + b);
        }

        $('#confusion_matrix tr td').removeClass("highlight");
        $('#confusion_matrix tr td[data-col="'+c+'"]').addClass("highlight");
        $('#confusion_matrix tr td[data-row="'+r+'"]').addClass("highlight");

    });
});
</script>
