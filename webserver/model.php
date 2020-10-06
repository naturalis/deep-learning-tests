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

    $t["h"]["h"] = "";
    foreach ($m as $cKey => $col)
    {
        $key = array_search($cKey, array_column($classes, "key"));
        $t["h"][] = $classes[$key]["key"] .". " . substr($classes[$key]["key"], 0, 5) . "&ellipsis";
    }

    foreach ($m as $cKey => $col)
    {
        $key = array_search($cKey, array_column($classes, "key"));
        $t[$cKey]["h"] = $classes[$key]["key"] . ". " . $classes[$key]["name"];
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

        var r = $(this).attr('data-row');
        var c = $(this).attr('data-col');
        var a = $('td[data-col="'+c+'"][data-row="h"]').html();
        var b = $('td[data-row="'+r+'"][data-col="h"]').html();

        var sum = 0;
        $('td[data-col="'+c+'"][data-row!="h"]').each(function()
        {
            sum += parseInt($(this).html());
        })


        $(this).attr("title",a + "\n" + b + "\n" + $(this).html() + " ("+ sum +")");
        // console.log(a + "\\" + b);

        $('#confusion_matrix tr td').removeClass("highlight");
        $('#confusion_matrix tr td[data-col="'+c+'"]').addClass("highlight");
        $('#confusion_matrix tr td[data-row="'+r+'"]').addClass("highlight");

    });
});
</script>
