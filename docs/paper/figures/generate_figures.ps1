# Reproducible, dependency-free manuscript figures. Values are transcribed only
# from the named audit artifacts; this script performs no inference or scoring.
Add-Type -AssemblyName System.Drawing
$out = $PSScriptRoot

function New-Canvas([int]$w, [int]$h) {
    $bmp = [System.Drawing.Bitmap]::new($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::White)
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
    return ,$bmp, $g
}
function Save-Canvas($bmp, $g, [string]$name) {
    $bmp.Save((Join-Path $out $name), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}
function Draw-Text($g, [string]$text, [float]$x, [float]$y, [float]$size, $color, [float]$width = 0) {
    $font = [System.Drawing.Font]::new('Arial', $size)
    $brush = [System.Drawing.SolidBrush]::new($color)
    if ($width -gt 0) { $g.DrawString($text, $font, $brush, [System.Drawing.RectangleF]::new($x, $y, $width, 200)) }
    else { $g.DrawString($text, $font, $brush, $x, $y) }
    $font.Dispose(); $brush.Dispose()
}
function Draw-BarChart($filename, $title, $labels, $series, [int]$maximum, $ylabel, [int]$axisFont = 18, [int]$legendFont = 17, [int]$leftMargin = 165, [int]$tickX = 105, [int]$ylabelWidth = 145) {
    $bmp, $g = New-Canvas 1300 680
    $black = [System.Drawing.Color]::FromArgb(40,40,40)
    Draw-Text $g $title 55 24 25 $black
    $left=$leftMargin; $top=100; $right=1240; $bottom=560; $plotH=$bottom-$top; $groupW=($right-$left)/$labels.Count
    $pen=[System.Drawing.Pen]::new($black,2); $g.DrawLine($pen,$left,$bottom,$right,$bottom); $g.DrawLine($pen,$left,$top,$left,$bottom)
    foreach($tick in @(0, [int]($maximum/2), $maximum)) { $y=$bottom-($tick/$maximum*$plotH); $grid=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(220,225,230),1); $g.DrawLine($grid,$left,$y,$right,$y); $grid.Dispose(); Draw-Text $g "$tick" $tickX ($y-$axisFont/1.5) $axisFont $black }
    Draw-Text $g $ylabel 8 290 $axisFont $black $ylabelWidth
    $colors=@([System.Drawing.Color]::FromArgb(76,120,168),[System.Drawing.Color]::FromArgb(245,133,24),[System.Drawing.Color]::FromArgb(84,162,75))
    for($s=0;$s -lt $series.Count;$s++) { $entry=$series[$s]; $barW=($groupW*0.52)/$series.Count; for($i=0;$i -lt $labels.Count;$i++){ $val=$entry.values[$i]; $x=$left+$i*$groupW+$groupW*.24+$s*$barW; $h=$val/$maximum*$plotH; $brush=[System.Drawing.SolidBrush]::new($colors[$s]); $g.FillRectangle($brush,$x,$bottom-$h,$barW-5,$h); $brush.Dispose(); Draw-Text $g "$val" ($x+4) ($bottom-$h-($axisFont+9)) ($axisFont-2) $black }; Draw-Text $g $entry.name 805 (60+$s*30) $legendFont $colors[$s] }
    for($i=0;$i -lt $labels.Count;$i++){ Draw-Text $g $labels[$i] ($left+$i*$groupW+$groupW*.20) 585 $axisFont $black ($groupW*.7) }
    $pen.Dispose(); Save-Canvas $bmp $g $filename
}

# Research evolution: conceptual figure; sources: docs/RESEARCH_AUDIT.md.
$bmp,$g=New-Canvas 1500 360; $black=[System.Drawing.Color]::FromArgb(40,40,40)
Draw-Text $g 'Research evolution: representations, candidates, then selection' 50 24 27 $black
$labels=@('Bounded symbolic libraries','Macro DSL + failure analysis','Evidence + RuleSpec','Native candidate search','Provenance-aware selection')
$fills=@([System.Drawing.Color]::FromArgb(220,234,247),[System.Drawing.Color]::FromArgb(252,228,214),[System.Drawing.Color]::FromArgb(226,240,217),[System.Drawing.Color]::FromArgb(255,242,204),[System.Drawing.Color]::FromArgb(217,225,242))
for($i=0;$i -lt 5;$i++){ $x=35+$i*292; $brush=[System.Drawing.SolidBrush]::new($fills[$i]); $g.FillRectangle($brush,$x,145,240,95); $brush.Dispose(); $pen=[System.Drawing.Pen]::new($black,2); $g.DrawRectangle($pen,$x,145,240,95); $pen.Dispose(); Draw-Text $g $labels[$i] ($x+12) 170 18 $black 215; if($i -lt 4){ $pen=[System.Drawing.Pen]::new($black,3); $g.DrawLine($pen,$x+240,192,$x+278,192); $pen.Dispose() } }
Save-Canvas $bmp $g 'research_evolution.png'

# Source: reports/public_reference_ablation_partial_v32.md, N=25 complete frozen30 tasks.
Draw-BarChart 'frozen30_recall_selection.png' 'Candidate recall does not directly determine selected success' @('A baseline','C search','D search + provenance') @(@{name='Any-of-K (diagnostic)';values=@(19,22,22)},@{name='Top-1';values=@(9,8,9)},@{name='Two attempts';values=@(11,9,13)}) 25 'Exact tasks (N=25)' 20 19
# Source: local/private artifacts/untouched60_scored.json. Frozen before inference and target access.
Draw-BarChart 'untouched60_ab.png' 'One untouched frozen A/B cohort (paired two-attempt p = 0.375)' @('Any-of-K (diagnostic)','Top-1','Two attempts') @(@{name='A baseline';values=@(30,18,18)},@{name='B provenance-aware';values=@(30,16,21)}) 60 'Exact tasks (N=60)' 20 19
# Source: local/private artifacts/untouched60_dynamic_queue_simulation.json. CPU-only replay.
Draw-BarChart 'scheduler_replay.png' 'Four-worker historical replay: unchanged inference semantics' @('Static','Dynamic queue') @(@{name='Modeled makespan (seconds)';values=@(7558,5564)}) 8500 'Modeled makespan (seconds)' 18 17 220 170 150
