<?php


    class HtmlClass
    {
        function header()
        {
            return <<< EOT
<html>
<link rel="stylesheet" type="text/css" href="style.css" >
<head>
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

        function list($l)
        {
            $b[] = "<ul>";
            foreach ($l as $val)
            {
                $b[] = "<li>$val</li>";
            }
            $b[] = "</ul>";

            return implode("\n", $b) . "\n";
        }

    }
