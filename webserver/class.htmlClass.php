<?php


    class HtmlClass
    {
        function header($fields=null)
        {
            return '<html>
<head>
'.(isset($fields["title"]) ? '<title>'.$fields["title"].'</title>' : '').'
    <link rel="stylesheet" type="text/css" href="style.css">
    <script type="text/javascript" src="jquery-3.5.1.min.js"></script>
    <script type="text/javascript" src="scripts.js"></script>
</head>
<body>
';
        }

        function h1($c,$id=null)
        {
            return "<h1".($id ? ' id="$id"' : "" ).">$c</h1>\n";
        }

        function h2($c,$id=null)
        {
            return "<h2".($id ? ' id="$id"' : "" ).">$c</h2>\n";
        }

        function h3($c,$id=null)
        {
            return "<h3".($id ? ' id="$id"' : "" ).">$c</h3>\n";
        }

        function p($c)
        {
            return "<p>$c</p>\n";
        }

        function span($c)
        {
            return "<span>$c</span>\n";
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
                    $datas = "";

                    foreach ($cell as $dKey => $data)
                    {
                        if (strpos($data, 'data-')==0)
                        {
                            $datas .= $dKey .'="'. $data .'" ';
                        }
                    }

                    $title = empty($cell["title"]) ? $cell["html"] : $cell["title"];
                    $b[] = "<td 
                                data-row='$rKey' 
                                data-col='$cKey'
                                $datas
                                title='" . $title . "'
                                class='" . $cell["class"] ."'>" . $cell["html"] ?? $cell . "</td>";
                }
                $b[] = "</tr>";
            }
            $b[] = "</table>";

            return implode("\n", $b) . "\n";
        }

        function select($l,$id=null)
        {
            $b[] = "<select id='$id' " . $a . ">";
            foreach ($l as $val)
            {
                $b[] = "<option value='$val[0]'>$val[1]</option>";
            }
            $b[] = "</select>";

            return implode("\n", $b) . "\n";
        }

        function button($value,$onclick=null,$id=null)
        {
            return "<input type='button' value='$value' onclick='$onclick' id='$id' />\n";
        }

        function image($src,$class)
        {
            return "<img src='$src' class='$class' alt='not seeing image? have you run webserver-set-symlink.sh?' />\n";
        }

    }
