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
            return "<h1>$c</h1>";
        }

        function h2($c)
        {
            return "<h2>$c</h2>";
        }

        function p($c)
        {
            return "<p>$c</p>";
        }

        function list($l)
        {
            $b[] = "<ul>";
            foreach ($l as $val)
            {
                $b[] = "<li>$val</li>";
            }
            $b[] = "</ul>";

            return implode("\n", $b);
        }

    }
