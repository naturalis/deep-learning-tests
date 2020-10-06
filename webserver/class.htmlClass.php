<?php


    class HtmlClass
    {
        function header()
        {
            return <<< EOT
<html>
<head>
    <link rel="stylesheet" type="text/css" href="style.css">
    <script type="text/javascript" src="jquery-3.5.1.min.js"></script>
</head>
<body>
EOT;
        }

        function h1($c)
        {
            return "<h1>$c</h1>\n";
        }

        function h2($c)
        {
            return "<h2>$c</h2>\n";
        }

        function p($c)
        {
            return "<p>$c</p>\n";
        }

        function list($l,$id=null)
        {
            $b[] = "<ul id='$id'>";
            foreach ($l as $val)
            {
                $b[] = "<li>$val</li>";
            }
            $b[] = "</ul>";

            return implode("\n", $b) . "\n";
        }

        function table($t,$id=null)
        {
            $b[] = "<table id='$id'>";
            foreach ($t as $rKey => $row)
            {
                $b[] = "<tr>";

                foreach ($row as $cKey => $cell)
                {
                    $title = empty($cell["title"]) ? ($cell["html"] ?? $cell) : $cell["title"];
                    $b[] = "<td data-row='$rKey' data-col='$cKey' title='" . $title . "'>" . $cell["html"] ?? $cell . "</td>";
                }
                $b[] = "</tr>";
            }
            $b[] = "</table>";

            return implode("\n", $b) . "\n";
        }

    }
