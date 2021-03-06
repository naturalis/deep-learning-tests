<?php

    include_once("class.baseClass.php");
    include_once("class.htmlClass.php");

    $base = new BaseClass;
    $base->setProjectRoot(getenv('PROJECT_ROOT'));
    $base->setProjectName(getenv('PROJECT_NAME'));
    $base->setImagesRoot(getenv('IMAGES_ROOT'));
    $base->setModels();
    $base->setModel($_GET["id"]);

    $l=[];
    foreach ($base->getModels() as $model)
    {        
        $l[]=
        [
            $model["model"],
            vsprintf("%s (acc: %s;  %s classes; %s)%s",
                [
                    $model["model"],
                    $model["accuracy"],
                    $model["dataset"]["class_count"],
                    $model["dataset"]["model_note"],
                    ($model["model"]==$_GET["id"] ? " 🡨" : "")
                ]
            ),

        ];
    }

    $size = $base->getModelSize();
    $dataset = $base->getDataset();
    $analysis = $base->getAnalysis();
    $classes = $base->getClasses();

    // $dataset["base_model"]
    // $dataset["class_image_minimum"]
    // $dataset["class_image_maximum"]
    // $dataset["use_class_balancing"]
    // $dataset["class_count"]
    // $dataset["class_count_before_maximum"]

    // $dataset["training_settings"]["validation_split"]
    // $dataset["training_settings"]["image_augmentation"]
    // $dataset["training_settings"]["batch_size"]

    // foreach ($dataset["training_phases"] as $key => $phase)
    // {

    // }

    // $phase["epochs"]
    // $phase["freeze_layers"]
    // $phase["optimizer"]
    // implode("; ",$phase["callbacks"])



    $html = new HtmlClass;

    echo $html->header(["title" => vsprintf("%s: %s [%s]",[$base->getProjectName(),$base->getModel(),$dataset["model_note"]])]);
    echo $html->h2(vsprintf("%s (%s)",[$base->getProjectName(),$base->getProjectRoot()]));
    echo $html->select($l,"models");
    echo $html->button("select",'openModelPage($("#models").val());');
    echo $html->span("<a href='index.php'>index</a>");
    echo $html->h2(vsprintf("%s: %s (%s)",[$base->getModel(),$dataset["model_note"],$dataset["created"]]));
    echo $html->p(vsprintf("%s (%s)",[$dataset["epochs_trained"],$dataset["training_time"]]));

    echo $html->h3("General analysis");

    $r=[];

    $r[] = [ [ "html" => "accuracy:" ], [ "html" => $analysis["classification_report"]["accuracy"] ] ];
    $r[] = [ [ "html" => "support:" ], [ "html" => $analysis["classification_report"]["macro avg"]["support"] ] ];
    $r[] = [ [ "html" => "" ], [ "html" => "" ] ];
    $r[] = [ [ "html" => "macro", "class" => "subheader" ], [ "html" => "" ] ];
    $r[] = [ [ "html" => "precision:" ], [ "html" => $analysis["classification_report"]["macro avg"]["precision"] ] ];
    $r[] = [ [ "html" => "recall:" ], [ "html" => $analysis["classification_report"]["macro avg"]["recall"] ] ];
    $r[] = [ [ "html" => "f1-score:" ], [ "html" => $analysis["classification_report"]["macro avg"]["f1-score"] ] ];
    $r[] = [ [ "html" => "" ], [ "html" => "" ] ];
    $r[] = [ [ "html" =>  "weighted", "class" => "subheader" ], [ "html" => "" ] ];
    $r[] = [ [ "html" =>  "precision:" ], [ "html" => $analysis["classification_report"]["weighted avg"]["precision"] ] ];
    $r[] = [ [ "html" => "recall:" ], [ "html" => $analysis["classification_report"]["weighted avg"]["recall"] ] ];
    $r[] = [ [ "html" => "f1-score:" ], [ "html" => $analysis["classification_report"]["weighted avg"]["f1-score"] ] ];


    echo $html->p($html->table($r,"analysis"));
    echo $html->h3("Classes");

    $c=[];

    $c[$key][] = [ "html" => "class" ];
    $c[$key][] = [ "html" => "support" ];
    $c[$key][] = [ "html" => "precision" ];
    $c[$key][] = [ "html" => "recall" ];
    $c[$key][] = [ "html" => "f1-score" ];
    $c[$key][] = [ "html" => "top_1" ];
    $c[$key][] = [ "html" => "top_3" ];
    $c[$key][] = [ "html" => "top_5" ];

    foreach ($classes as $cKey => $class)
    {
        if (!isset($class["key"]))
        {
            continue;
        }

        $key = array_search($cKey, array_column($analysis["top_k_per_class"], "class"));

        $c[$key][] = [ "html" => $class["name"], "data-hash" => md5($class["name"]) ];
        $c[$key][] = [ "html" => $class["support"] ];
        // $c[$key][] = [ "html" => $analysis["classification_report"][$class["key"]]["support"] ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["f1-score"],2) ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["precision"],2) ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["recall"],2) ];
        $c[$key][] = [ "html" => round($analysis["classification_report"][$class["key"]]["f1-score"],2) ];

        $c[$key][] = [ "html" => round($analysis["top_k_per_class"][$class["key"]]["top_1"] / $class["support"],2) ];
        $c[$key][] = [ "html" => round($analysis["top_k_per_class"][$class["key"]]["top_3"] / $class["support"],2) ];
        $c[$key][] = [ "html" => round($analysis["top_k_per_class"][$class["key"]]["top_5"] / $class["support"],2) ];
    }


    echo $html->p($html->table($c,"classes"));

    // confusion matrix
    echo $html->h3("Confusion matrix");
    echo $html->image($base->getModelImagePath("confusion_matrix.png"),"confusion-matrix");

    $m = $analysis["confusion_matrix"];

    $t=[];


    $t["h"]["h"] = [ "html" => "" ];
    foreach ($m as $rKey => $row)
    {
        $key = array_search($rKey, array_column($classes, "key"));
        $t["h"][] = [ 
            "html" => $classes[$key]["key"],
            "title" => $classes[$key]["name"],
            "data-hash" => md5($classes[$key]["name"])
        ];
    }

    foreach ($m as $rKey => $row)
    {
        $key = array_search($rKey, array_column($classes, "key"));
        $t[$rKey]["h"] = [
            "html" => $classes[$key]["key"] . ". " . $classes[$key]["name"],
            "title" => $classes[$key]["name"],
            "data-hash" => md5($classes[$key]["name"])
        ];

        foreach ($row as $cKey => $col)
        {
            $t[$rKey][$cKey] =  [ "html" => $col ];
        }
    }

    echo $html->p($html->table($t,"confusion_matrix"));

    // echo "<pre>";
    // print_r($classes);
    // print_r($analysis);
    // print_r($dataset);

?>
<script type="text/javascript">
var prevSortedIndex = -1;
var prevSortedAsc = true;

function sortTable(table,index,ele)
{

    elements=[];
    header = "";

    $('#classes tr td:nth-child('+(index+1)+')').each(function(a,b)
    {
        if (a==0)
        {
            header = $(b).parent('tr').html();
            return;
        }

        if (isNaN($(this).html()))
        {
            elements.push({ "index": a, "value" : $(this).html(), "html" : $(b).parent('tr').html() })
        }
        else
        {
            elements.push({ "index": a, "value" : parseFloat($(this).html()), "html" : $(b).parent('tr').html() })
        }       
    });

    elements.sort(function(a, b)
    {
        if (prevSortedIndex==index)
        {
            asc = !prevSortedAsc;
        }
        else
        {
            asc = isNaN(a.value);
        }

        if (a.value==b.value)
        {
            return 0;
        }
        else
        {
            return (a.value > b.value ? 1 : -1) * (asc ? 1 : -1 );
        }
    });    

    // console.dir(elements);

    prevSortedIndex = index;
    prevSortedAsc = asc;

    var rows = elements.map(function(a){ return "<tr>" + a.html + "</tr>"; });
    rows.unshift("<tr>" + header + "</tr>")

    $(table).html(rows.join("\n"));
    bootstrap();
}

function bootstrap()
{
    $('#classes tr:first td').each(function(index,ele)
    {
        $(this).on('click',function()
        {
            sortTable($('#classes'),index,ele);
        })
    });

    $("tr td").mouseover(function()
    {
        var h = $(this).attr('data-hash');
        $('td').removeClass('highlight-class');
        $('td[data-hash="'+h+'"]').addClass('highlight-class');
        // $('td[data-hash="'+h+'"]').parent().find('td').addClass('highlight-class');
    });    
}

$( document ).ready(function()
{
    $("#confusion_matrix tr td").mouseover(function()
    {
        var r = $(this).attr('data-row');
        var c = $(this).attr('data-col');
        var a = $('td[data-col="'+c+'"][data-row="h"]').attr("title");
        var b = $('td[data-row="'+r+'"][data-col="h"]').attr("title");

        var sum = 0;
        $('#confusion_matrix tr td[data-col="'+c+'"][data-row!="h"]').each(function()
        {
            sum += parseInt($(this).html().trim());
            console.log(sum,parseInt($(this).html()),$(this).html());
        })

        if (r!="h" && c!="h")
        {
            $(this).attr("title",$(this).html() + " (of " + sum +")\n" + a + "\nidentified as\n" + b);
            // console.log(a + "\\" + b);
        }

        $('#confusion_matrix tr td').removeClass("highlight");
        $('#confusion_matrix tr td[data-col="'+c+'"]').addClass("highlight");
        $('#confusion_matrix tr td[data-row="'+r+'"]').addClass("highlight");
    });

    bootstrap();
});
</script>
