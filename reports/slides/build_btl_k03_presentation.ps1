$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path '.').Path
$slidesDir = Join-Path $projectRoot 'reports\slides'
$outputPath = Join-Path $slidesDir 'BTL-K03-EDA-PREPROCESSING-18-SLIDES.pptx'
$previewPath = Join-Path $projectRoot 'pytest\pptx_preview'

$figSession = Join-Path $projectRoot 'results\figures\eda_session_length_histogram_ecdf.png'
$figLorenz = Join-Path $projectRoot 'results\figures\eda_interaction_lorenz_curve.png'
$figThreshold = Join-Path $projectRoot 'results\figures\eda_frequency_threshold_tradeoff.png'
$figCategory = Join-Path $projectRoot 'results\figures\eda_category_structure_and_transition.png'
$figTemporal = Join-Path $projectRoot 'results\figures\preprocessing_temporal_split.png'
$figAttrition = Join-Path $projectRoot 'results\figures\preprocessing_attrition_and_groups.png'

function Color([int]$r, [int]$g, [int]$b) {
    return $r + 256 * $g + 65536 * $b
}

$C = @{
    Navy = Color 11 31 51
    Navy2 = Color 20 49 82
    Blue = Color 47 107 255
    Teal = Color 22 179 163
    Orange = Color 255 159 67
    Red = Color 228 87 86
    Green = Color 44 160 101
    Ink = Color 23 43 77
    Gray = Color 96 112 137
    Mid = Color 198 208 221
    Light = Color 244 247 251
    PaleBlue = Color 232 240 255
    PaleTeal = Color 228 248 245
    PaleOrange = Color 255 244 228
    White = Color 255 255 255
}

function Add-Rect($slide, $x, $y, $w, $h, $fill, $radius = $false, $line = $null, $transparency = 0) {
    $shapeType = if ($radius) { 5 } else { 1 }
    $shape = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Fill.Transparency = $transparency
    if ($null -eq $line) {
        $shape.Line.Visible = 0
    } else {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = $line
        $shape.Line.Weight = 1
    }
    return $shape
}

function Add-Text($slide, [string]$text, $x, $y, $w, $h, $size, $color, $bold = $false, $align = 1, $font = 'Aptos') {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.Fill.Visible = 0
    $shape.Line.Visible = 0
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.TextRange.Font.Name = $font
    $shape.TextFrame2.TextRange.Font.Size = [single]$size
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    $shape.TextFrame2.TextRange.Font.Bold = if ($bold) { -1 } else { 0 }
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Background($slide, $color) {
    [void](Add-Rect $slide 0 0 960 540 $color)
}

function Add-Footer($slide, [int]$number, [string]$section) {
    [void](Add-Rect $slide 46 506 868 1 $C.Mid)
    [void](Add-Text $slide "BTL-K03  /  $section  /  DIGINETICA" 48 514 600 14 8.5 $C.Gray $true)
    [void](Add-Text $slide ("{0:D2} / 18" -f $number) 818 512 94 16 9 $C.Gray $true 3)
}

function Add-Header($slide, [int]$number, [string]$title, [string]$message, [string]$section = 'NGHIÊN CỨU') {
    Add-Background $slide $C.White
    [void](Add-Rect $slide 0 0 14 540 $C.Blue)
    [void](Add-Text $slide $title 48 28 850 44 27 $C.Navy $true)
    [void](Add-Text $slide $message 49 78 850 30 13.5 $C.Gray $false)
    Add-Footer $slide $number $section
}

function Add-Bullets($slide, [string[]]$items, $x, $y, $w, $h, $size = 15, $color = $null, $accent = $null) {
    if ($null -eq $color) { $color = $C.Ink }
    if ($null -eq $accent) { $accent = $C.Blue }
    $lineHeight = $h / [Math]::Max($items.Count, 1)
    for ($i = 0; $i -lt $items.Count; $i++) {
        [void](Add-Rect $slide $x ($y + $i * $lineHeight + 5) 7 7 $accent $true)
        [void](Add-Text $slide $items[$i] ($x + 18) ($y + $i * $lineHeight) ($w - 18) ($lineHeight - 2) $size $color $false)
    }
}

function Add-Kpi($slide, [string]$value, [string]$label, $x, $y, $w, $fill, $accent) {
    [void](Add-Rect $slide $x $y $w 94 $fill $true)
    [void](Add-Rect $slide $x $y 7 94 $accent $true)
    [void](Add-Text $slide $value ($x + 18) ($y + 15) ($w - 30) 34 25 $C.Navy $true)
    [void](Add-Text $slide $label ($x + 18) ($y + 55) ($w - 30) 24 10.5 $C.Gray $true)
}

function Add-Tag($slide, [string]$text, $x, $y, $w, $fill, $fontColor) {
    [void](Add-Rect $slide $x $y $w 28 $fill $true)
    [void](Add-Text $slide $text ($x + 8) ($y + 6) ($w - 16) 15 9.5 $fontColor $true 2)
}

function Add-PictureFit($slide, [string]$path, $x, $y, $w, $h) {
    $pic = $slide.Shapes.AddPicture($path, 0, -1, 0, 0, -1, -1)
    $scale = [Math]::Min($w / $pic.Width, $h / $pic.Height)
    $pic.Width = $pic.Width * $scale
    $pic.Height = $pic.Height * $scale
    $pic.Left = $x + ($w - $pic.Width) / 2
    $pic.Top = $y + ($h - $pic.Height) / 2
    return $pic
}

function Add-Arrow($slide, $x1, $y1, $x2, $y2, $color, $weight = 2.5) {
    $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = [single]$weight
    $line.Line.EndArrowheadStyle = 3
    return $line
}

function Add-SectionLabel($slide, [string]$text, $x, $y, $w, $color) {
    [void](Add-Text $slide $text $x $y $w 18 9.5 $color $true)
}

$ppt = $null
$presentation = $null

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = -1
    try { $ppt.WindowState = 2 } catch {}
    $presentation = $ppt.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    # Slide 1
    $s = $presentation.Slides.Add(1, 12)
    Add-Background $s $C.Navy
    [void](Add-Rect $s 0 0 18 540 $C.Blue)
    Add-Tag $s 'BTL-K03  •  DIGINETICA' 54 42 210 $C.Blue $C.White
    [void](Add-Text $s 'Thiên lệch độ phổ biến và hình học không gian embedding' 54 102 795 104 33 $C.White $true)
    [void](Add-Text $s 'trong mô hình Session-Based Recommendation' 55 214 780 42 20 $C.Mid $false)
    [void](Add-Text $s 'EDA • Tiền xử lý • Giao thức cold-start 20%' 55 302 650 28 14 $C.Teal $true)
    $x = 55
    foreach ($item in @(@('GRU4Rec',$C.PaleBlue,$C.Blue),@('NARM',$C.PaleTeal,$C.Teal),@('SASRec',$C.PaleOrange,$C.Orange))) {
        [void](Add-Rect $s $x 362 142 48 $item[1] $true)
        [void](Add-Text $s $item[0] $x 376 142 22 13 $C.Navy $true 2)
        $x += 158
    }
    [void](Add-Text $s 'Trạng thái: hoàn thành notebook 01 và 02' 55 455 530 20 10.5 $C.Mid $true)
    [void](Add-Text $s '01 / 18' 826 490 82 18 10 $C.Mid $true 3)

    # Slide 2
    $s = $presentation.Slides.Add(2, 12)
    Add-Header $s 2 'Từ chuỗi hành vi đến dự đoán item tiếp theo' 'SBR dự đoán item tiếp theo từ hành vi trong session hiện tại.' 'MỞ ĐẦU'
    Add-SectionLabel $s 'MỘT SESSION' 50 126 170 $C.Blue
    $labels = @('A','B','C','D')
    $fills = @($C.PaleBlue,$C.PaleBlue,$C.PaleTeal,$C.PaleOrange)
    for ($i=0; $i -lt 4; $i++) {
        $x = 70 + $i*150
        [void](Add-Rect $s $x 170 82 82 $fills[$i] $true)
        [void](Add-Text $s $labels[$i] $x 190 82 35 24 $C.Navy $true 2)
        if ($i -lt 3) { [void](Add-Arrow $s ($x+92) 211 ($x+138) 211 $C.Blue 2.2) }
    }
    [void](Add-Text $s 'Lịch sử quan sát' 74 268 370 22 11 $C.Gray $true 2)
    [void](Add-Text $s 'Target' 523 268 95 22 11 $C.Orange $true 2)
    [void](Add-Rect $s 660 138 240 258 $C.Light $true)
    [void](Add-Text $s 'Tạo mẫu next-item' 684 158 190 26 15 $C.Navy $true 2)
    Add-Bullets $s @('[A]  →  B','[A, B]  →  C','[A, B, C]  →  D') 690 210 180 128 15 $C.Ink $C.Teal
    [void](Add-Text $s 'Thứ tự click sai → quan hệ chuyển tiếp sai' 76 357 515 31 17 $C.Red $true)
    [void](Add-Text $s 'Vì vậy kiểm tra và khôi phục thứ tự session là bước bắt buộc.' 76 399 522 42 13 $C.Gray $false)

    # Slide 3
    $s = $presentation.Slides.Add(3, 12)
    Add-Header $s 3 'Mô hình có đang khuếch đại lợi thế của item phổ biến?' 'Accuracy cao chưa đủ; cần đo cách cơ hội hiển thị được phân phối.' 'VẤN ĐỀ'
    $cx=310; $cy=285
    $nodes=@(
        @('Item đã phổ biến',180,145,$C.PaleOrange,$C.Orange),
        @('Được gợi ý nhiều',500,145,$C.PaleBlue,$C.Blue),
        @('Nhận thêm tương tác',500,330,$C.PaleTeal,$C.Teal),
        @('Càng phổ biến',180,330,$C.Light,$C.Red)
    )
    foreach($n in $nodes){ [void](Add-Rect $s $n[1] $n[2] 200 70 $n[3] $true); [void](Add-Text $s $n[0] $n[1] ($n[2]+22) 200 25 14 $C.Navy $true 2) }
    [void](Add-Arrow $s 390 180 490 180 $C.Blue)
    [void](Add-Arrow $s 600 220 600 320 $C.Teal)
    [void](Add-Arrow $s 500 365 390 365 $C.Red)
    [void](Add-Arrow $s 280 330 280 220 $C.Orange)
    [void](Add-Rect $s 742 145 170 255 $C.Navy $true)
    [void](Add-Text $s 'Cần đo đồng thời' 760 169 134 36 13 $C.White $true 2)
    [void](Add-Text $s '01' 765 226 34 28 19 $C.Teal $true)
    [void](Add-Text $s 'Độ chính xác' 807 230 92 22 12 $C.White $true)
    [void](Add-Text $s '02' 765 282 34 28 19 $C.Orange $true)
    [void](Add-Text $s 'Độ tập trung Top-20' 807 279 92 42 12 $C.White $true)
    [void](Add-Text $s '03' 765 347 34 28 19 $C.Blue $true)
    [void](Add-Text $s 'Hình học embedding' 807 342 92 42 12 $C.White $true)

    # Slide 4
    $s = $presentation.Slides.Add(4, 12)
    Add-Header $s 4 'Kết nối dữ liệu, embedding và đầu ra gợi ý' 'Phần sâu của đề tài là liên hệ ba đại lượng ở cấp item.' 'Ý TƯỞNG'
    $centers=@(@(218,245,'TRAIN FREQUENCY',$C.Blue,$C.PaleBlue),@(480,365,'EMBEDDING NORM',$C.Teal,$C.PaleTeal),@(742,245,'REC COUNT',$C.Orange,$C.PaleOrange))
    foreach($n in $centers){ [void](Add-Rect $s ($n[0]-108) ($n[1]-42) 216 84 $n[4] $true); [void](Add-Text $s $n[2] ($n[0]-100) ($n[1]-10) 200 22 13 $C.Navy $true 2) }
    [void](Add-Arrow $s 326 245 630 245 $C.Blue 2)
    [void](Add-Arrow $s 680 282 555 344 $C.Orange 2)
    [void](Add-Arrow $s 405 344 280 282 $C.Teal 2)
    [void](Add-Text $s 'Item phổ biến có được gợi ý nhiều hơn?' 350 177 260 24 11 $C.Gray $true 2)
    [void](Add-Text $s 'Norm có đi cùng lượt gợi ý?' 555 316 205 22 11 $C.Gray $true 2)
    [void](Add-Text $s 'Frequency có đi cùng norm?' 198 316 205 22 11 $C.Gray $true 2)
    [void](Add-Rect $s 170 436 620 38 $C.Light $true)
    [void](Add-Text $s 'Kết luận mối liên hệ — không tự động khẳng định quan hệ nhân quả' 190 447 580 18 11.5 $C.Red $true 2)

    # Slide 5
    $s = $presentation.Slides.Add(5, 12)
    Add-Header $s 5 'Diginetica: đủ lớn, có session và có category' 'Dataset đáp ứng cả bài toán next-item lẫn phân tích long-tail.' 'DỮ LIỆU'
    Add-Kpi $s '1.235.380' 'INTERACTION' 50 132 198 $C.PaleBlue $C.Blue
    Add-Kpi $s '310.324' 'SESSION' 266 132 198 $C.PaleTeal $C.Teal
    Add-Kpi $s '122.993' 'ITEM CÓ TƯƠNG TÁC' 482 132 198 $C.PaleOrange $C.Orange
    Add-Kpi $s '1.217' 'CATEGORY' 698 132 198 $C.Light $C.Red
    $files=@(
        @('train-item-views.csv','Chuỗi session • item • thời gian','1.235.380 dòng'),
        @('product-categories.csv','Ánh xạ item → category','184.047 dòng'),
        @('products.csv','Kiểm tra catalog và metadata','184.047 dòng')
    )
    for($i=0;$i -lt 3;$i++){
        $x=50+$i*292
        [void](Add-Rect $s $x 270 270 126 $C.Light $true $C.Mid)
        [void](Add-Text $s $files[$i][0] ($x+16) 288 238 22 12 $C.Navy $true)
        [void](Add-Text $s $files[$i][1] ($x+16) 324 238 30 11 $C.Gray $false)
        [void](Add-Text $s $files[$i][2] ($x+16) 365 238 18 10 $C.Blue $true)
    }
    [void](Add-Text $s '152 ngày có quan sát  •  01/01–01/06/2016  •  liên kết qua itemId' 50 435 846 26 13 $C.Ink $true 2)

    # Slide 6
    $s = $presentation.Slides.Add(6, 12)
    Add-Header $s 6 'EDA để ra quyết định, không chỉ để mô tả' 'Notebook 01 trả lời các câu hỏi phải giải quyết trước khi huấn luyện.' 'EDA'
    $cards=@(
        @('01','Dữ liệu có sạch?','Schema • missing • duplicate',$C.Blue,$C.PaleBlue),
        @('02','Chuỗi có đúng?','Thứ tự • session • click lặp',$C.Teal,$C.PaleTeal),
        @('03','Có long-tail?','Frequency • Gini • concentration',$C.Orange,$C.PaleOrange),
        @('04','Category có ý nghĩa?','Transition cùng/khác category',$C.Red,$C.Light),
        @('05','Lọc mất gì?','Threshold sensitivity • attrition',$C.Green,$C.PaleTeal)
    )
    for($i=0;$i -lt 5;$i++){
        $x=50+($i%3)*292; $y=135+[Math]::Floor($i/3)*145
        [void](Add-Rect $s $x $y 270 116 $cards[$i][4] $true)
        [void](Add-Text $s $cards[$i][0] ($x+16) ($y+15) 42 26 18 $cards[$i][3] $true)
        [void](Add-Text $s $cards[$i][1] ($x+62) ($y+17) 190 24 13 $C.Navy $true)
        [void](Add-Text $s $cards[$i][2] ($x+62) ($y+53) 190 38 10.5 $C.Gray $false)
    }
    [void](Add-Rect $s 634 280 262 116 $C.Navy $true)
    [void](Add-Text $s 'KẾT QUẢ' 652 298 225 18 10 $C.Teal $true 2)
    [void](Add-Text $s 'Quy tắc preprocessing có bằng chứng' 654 330 222 46 15 $C.White $true 2)

    # Slide 7
    $s = $presentation.Slides.Add(7, 12)
    Add-Header $s 7 'Dữ liệu liên kết tốt nhưng thứ tự raw không đáng tin cậy' 'Rủi ro lớn nhất nằm ở chuỗi thời gian, không phải missing hay duplicate.' 'EDA'
    Add-Kpi $s '0' 'DUPLICATE CHÍNH XÁC' 50 132 194 $C.Light $C.Green
    Add-Kpi $s '0' 'ITEM THIẾU CATEGORY' 260 132 194 $C.Light $C.Green
    Add-Kpi $s '69,81%' 'USER ID BỊ THIẾU' 470 132 194 $C.PaleOrange $C.Orange
    Add-Kpi $s '58,21%' 'SESSION SAI THỨ TỰ RAW' 680 132 216 $C.PaleBlue $C.Blue
    [void](Add-Text $s 'TRƯỚC SORT' 72 286 175 18 9.5 $C.Red $true)
    foreach($pair in @(@('C',75),@('A',160),@('B',245))){ [void](Add-Rect $s $pair[1] 322 56 56 $C.Light $true); [void](Add-Text $s $pair[0] $pair[1] 339 56 24 16 $C.Navy $true 2) }
    [void](Add-Text $s 'SAU SORT' 397 286 175 18 9.5 $C.Teal $true)
    foreach($pair in @(@('A',400),@('B',485),@('C',570))){ [void](Add-Rect $s $pair[1] 322 56 56 $C.PaleTeal $true); [void](Add-Text $s $pair[0] $pair[1] 339 56 24 16 $C.Navy $true 2) }
    [void](Add-Arrow $s 318 350 376 350 $C.Blue 3)
    [void](Add-Rect $s 682 286 214 121 $C.Navy $true)
    [void](Add-Text $s 'QUY TẮC CHÍNH THỨC' 700 304 180 18 9 $C.Teal $true 2)
    [void](Add-Text $s "sessionId`ntimeframe`nrow_order" 716 334 150 58 14 $C.White $true 2)
    [void](Add-Text $s '4 nhóm trùng timeframe → giữ event, dùng row_order để phá hòa' 70 432 810 30 11.5 $C.Gray $false 2)

    # Slide 8
    $s = $presentation.Slides.Add(8, 12)
    Add-Header $s 8 'Session ngắn chiếm đa số, click lặp vẫn là tín hiệu' 'Độ dài phiên quyết định khả năng tạo target và trọng số mẫu.' 'EDA'
    [void](Add-PictureFit $s $figSession 42 125 570 310)
    Add-Kpi $s '3' 'MEDIAN SESSION LENGTH' 640 130 256 $C.PaleBlue $C.Blue
    Add-Kpi $s '29,23%' 'SESSION CHỈ 1 CLICK' 640 238 256 $C.PaleOrange $C.Orange
    Add-Kpi $s '13,62%' 'EVENT LÀ CLICK LẶP' 640 346 256 $C.PaleTeal $C.Teal
    [void](Add-Text $s 'Quyết định: bỏ session length 1 • giữ click lặp • theo dõi đóng góp của session dài' 52 457 844 26 11.5 $C.Ink $true 2)

    # Slide 9
    $s = $presentation.Slides.Add(9, 12)
    Add-Header $s 9 'Diginetica có long-tail và mức tập trung rất mạnh' 'Dataset phù hợp nghiên cứu bias — nhưng chưa phải bằng chứng bias của mô hình.' 'EDA'
    [void](Add-PictureFit $s $figLorenz 52 124 400 345)
    Add-Kpi $s '0,726' 'INPUT-DATA GINI' 500 132 188 $C.PaleOrange $C.Orange
    Add-Kpi $s '63,80%' 'ITEM DƯỚI 5 LẦN' 708 132 188 $C.PaleBlue $C.Blue
    Add-Kpi $s '77,54%' 'INTERACTION DO TOP 20% ITEM' 500 252 396 $C.PaleTeal $C.Teal
    [void](Add-Rect $s 500 374 396 70 $C.Navy $true)
    [void](Add-Text $s 'Gini 0,726 ≠ Recommendation Gini' 524 392 348 22 15 $C.White $true 2)
    [void](Add-Text $s 'Mô hình chưa được huấn luyện ở bước EDA.' 524 420 348 15 9.5 $C.Mid $false 2)

    # Slide 10
    $s = $presentation.Slides.Add(10, 12)
    Add-Header $s 10 'Category có tín hiệu mạnh, tuổi item cần được lưu ý' 'Category hỗ trợ cosine analysis; item age giới hạn diễn giải nhân quả.' 'EDA'
    [void](Add-PictureFit $s $figCategory 42 128 650 250)
    [void](Add-Rect $s 716 130 180 112 $C.PaleTeal $true)
    [void](Add-Text $s '87,45%' 734 151 144 34 25 $C.Teal $true 2)
    [void](Add-Text $s "cùng category`nsau bỏ self-repeat" 734 193 144 32 10 $C.Gray $true 2)
    [void](Add-Rect $s 716 258 180 112 $C.Light $true)
    [void](Add-Text $s '1,08%' 734 279 144 34 25 $C.Red $true 2)
    [void](Add-Text $s 'baseline ngẫu nhiên' 734 324 144 20 10 $C.Gray $true 2)
    [void](Add-Rect $s 50 405 846 62 $C.PaleOrange $true)
    [void](Add-Text $s 'Item age ↔ frequency: Spearman ρ = 0,504' 70 419 360 25 13 $C.Navy $true)
    [void](Add-Text $s '→ Phân tích embedding chỉ kết luận “liên hệ”, không kết luận nhân quả.' 438 419 430 30 11.5 $C.Red $true)

    # Slide 11
    $s = $presentation.Slides.Add(11, 12)
    Add-Header $s 11 'Lọc item giúp học ổn định hơn nhưng cắt mạnh long-tail' 'Ngưỡng 5 là baseline có trade-off, không phải giá trị tối ưu đã chứng minh.' 'EDA'
    [void](Add-PictureFit $s $figThreshold 38 118 625 360)
    Add-Kpi $s '42.171' 'ITEM CÒN Ở NGƯỠNG 5' 690 132 206 $C.PaleBlue $C.Blue
    Add-Kpi $s '80,07%' 'EVENT ĐƯỢC GIỮ' 690 244 206 $C.PaleTeal $C.Teal
    Add-Kpi $s '34,29%' 'ITEM RAW ĐƯỢC GIỮ' 690 356 206 $C.PaleOrange $C.Orange
    [void](Add-Text $s 'Gini giảm sau lọc chủ yếu do catalog tail bị cắt.' 690 456 206 32 9.5 $C.Red $true 2)

    # Slide 12
    $s = $presentation.Slides.Add(12, 12)
    Add-Header $s 12 'Mỗi phát hiện EDA được chuyển thành một quyết định xử lý' 'Notebook 02 là lời giải có hệ thống cho các rủi ro đã tìm thấy.' 'GIẢI PHÁP'
    $rows=@(
        @('58,21% sai thứ tự','Stable sort theo timeframe + row_order',$C.Blue),
        @('29,23% session 1 click','Loại session không tạo được target',$C.Orange),
        @('13,62% click lặp','Giữ lại tín hiệu quay lại item',$C.Teal),
        @('260 session qua đêm','Giữ nguyên; chia theo ngày cuối session',$C.Red),
        @('Long-tail rất thưa','Frequency ≥ 5 + fixed-point filtering',$C.Green),
        @('Nguy cơ leakage','Popularity chỉ tính từ final train',$C.Blue)
    )
    for($i=0;$i -lt $rows.Count;$i++){
        $y=128+$i*56
        [void](Add-Rect $s 50 $y 305 44 $C.Light $true)
        [void](Add-Text $s $rows[$i][0] 67 ($y+12) 270 20 11.5 $C.Ink $true)
        [void](Add-Arrow $s 373 ($y+22) 420 ($y+22) $rows[$i][2] 2)
        [void](Add-Rect $s 438 $y 458 44 $C.PaleBlue $true)
        [void](Add-Text $s $rows[$i][1] 456 ($y+12) 422 20 11.5 $C.Navy $true)
    }

    # Slide 13
    $s = $presentation.Slides.Add(13, 12)
    Add-Header $s 13 'Học từ quá khứ, đánh giá trên tương lai' 'Temporal split ở cấp session giúp tránh temporal leakage.' 'PREPROCESSING'
    [void](Add-PictureFit $s $figTemporal 42 120 585 300)
    $segments=@(
        @('TRAIN','01/01–18/05','990.708 event',$C.Blue,$C.PaleBlue),
        @('VALIDATION','19/05–25/05','65.533 event',$C.Orange,$C.PaleOrange),
        @('TEST','26/05–01/06','88.445 event',$C.Teal,$C.PaleTeal)
    )
    for($i=0;$i -lt 3;$i++){
        $y=130+$i*100
        [void](Add-Rect $s 655 $y 241 80 $segments[$i][4] $true)
        [void](Add-Text $s $segments[$i][0] 672 ($y+12) 210 18 10 $segments[$i][3] $true)
        [void](Add-Text $s $segments[$i][1] 672 ($y+34) 210 19 13 $C.Navy $true)
        [void](Add-Text $s $segments[$i][2] 672 ($y+58) 210 14 9 $C.Gray $false)
    }
    [void](Add-Text $s 'Toàn bộ session được gán theo ngày event cuối • cửa sổ 7 ngày là baseline, chưa chứng minh tối ưu' 54 450 842 30 10.5 $C.Red $true 2)

    # Slide 14
    $s = $presentation.Slides.Add(14, 12)
    Add-Header $s 14 'Lọc lặp đến khi item và session cùng ổn định' 'Xóa item và xóa session tác động qua lại, nên một lượt lọc là chưa đủ.' 'PREPROCESSING'
    $steps=@(
        @('Lọc item','frequency ≥ 5',$C.PaleBlue,$C.Blue),
        @('Lọc session','length ≥ 2',$C.PaleTeal,$C.Teal),
        @('Kiểm tra lại','còn thay đổi?',$C.PaleOrange,$C.Orange)
    )
    for($i=0;$i -lt 3;$i++){
        $x=55+$i*210
        [void](Add-Rect $s $x 150 170 82 $steps[$i][2] $true)
        [void](Add-Text $s $steps[$i][0] ($x+12) 168 146 22 13 $C.Navy $true 2)
        [void](Add-Text $s $steps[$i][1] ($x+12) 199 146 18 10 $C.Gray $true 2)
        if($i -lt 2){ [void](Add-Arrow $s ($x+178) 191 ($x+202) 191 $steps[$i][3] 2.2) }
    }
    [void](Add-Arrow $s 565 245 140 265 $C.Orange 2)
    [void](Add-Text $s 'lặp đến fixed point' 294 247 160 20 10 $C.Orange $true 2)
    [void](Add-Rect $s 686 138 210 260 $C.Navy $true)
    [void](Add-Text $s 'ATTRITION TRƯỚC COLD' 705 160 172 18 9 $C.Teal $true 2)
    [void](Add-Text $s '114.768' 710 202 162 34 25 $C.White $true 2)
    [void](Add-Text $s 'item sau temporal split' 710 240 162 18 9.5 $C.Mid $false 2)
    [void](Add-Text $s '↓' 750 274 82 30 22 $C.Orange $true 2)
    [void](Add-Text $s '38.237' 710 313 162 34 25 $C.White $true 2)
    [void](Add-Text $s 'item sau fixed-point' 710 351 162 18 9.5 $C.Mid $false 2)
    [void](Add-Text $s '841.402 event • 174.691 session • 666.711 prefix-target' 58 420 598 28 12 $C.Ink $true 2)

    # Slide 15
    $s = $presentation.Slides.Add(15, 12)
    Add-Header $s 15 'Controlled cold để quan sát embedding chưa được huấn luyện' 'Cold được tạo có kiểm soát và tách khỏi cold xuất hiện tự nhiên theo thời gian.' 'PREPROCESSING'
    [void](Add-Text $s 'CATALOG TRƯỚC HOLDOUT' 55 131 260 18 9.5 $C.Gray $true)
    [void](Add-Rect $s 55 165 540 72 $C.PaleBlue $true)
    [void](Add-Rect $s 55 165 432 72 $C.Blue $true)
    [void](Add-Rect $s 487 165 108 72 $C.Orange $true)
    [void](Add-Text $s '80% WARM' 196 188 180 24 14 $C.White $true 2)
    [void](Add-Text $s '20% COLD' 494 188 94 24 12 $C.White $true 2)
    [void](Add-Text $s '38.237 item  →  7.647 controlled cold  •  seed 2024  •  phân tầng category' 55 251 540 24 12 $C.Ink $true 2)
    [void](Add-Rect $s 55 303 260 126 $C.PaleOrange $true)
    [void](Add-Text $s 'CONTROLLED COLD' 73 319 224 20 11 $C.Orange $true 2)
    Add-Bullets $s @('Chủ động holdout khỏi train','Giữ ID + category','629/629 category còn warm peer') 76 349 218 66 9.8 $C.Ink $C.Orange
    [void](Add-Rect $s 335 303 260 126 $C.PaleTeal $true)
    [void](Add-Text $s 'NATURAL TEMPORAL COLD' 353 319 224 20 11 $C.Teal $true 2)
    Add-Bullets $s @('Tự xuất hiện sau mốc train','6.700 item','Theo dõi riêng, không trộn nhãn') 356 349 218 66 9.8 $C.Ink $C.Teal
    [void](Add-Rect $s 630 150 266 279 $C.Navy $true)
    [void](Add-Text $s 'ĐIỀU KIỆN BẮT BUỘC' 650 173 226 18 9.5 $C.Teal $true 2)
    Add-Bullets $s @('Không positive target','Không negative sampling','Không nhận gradient','Lưu embedding khởi tạo') 657 218 210 142 12 $C.White $C.Orange
    [void](Add-Text $s 'Cold chỉ dùng minh họa trong giao thức hiện tại.' 651 383 224 28 9.5 $C.Mid $true 2)

    # Slide 16
    $s = $presentation.Slides.Add(16, 12)
    Add-Header $s 16 'Một bộ dữ liệu chung đã sẵn sàng cho ba mô hình' 'Final split đủ lớn và popularity được tính chỉ từ final train.' 'KẾT QUẢ'
    $tableX=48; $tableY=132; $colW=@(155,112,112,112)
    $headers=@('TẬP','EVENT','SESSION','ITEM')
    $rows=@(
        @('Train','649.632','150.783','29.936'),
        @('Validation warm','39.450','9.622','12.227'),
        @('Test warm','52.792','12.791','13.847')
    )
    $x=$tableX
    for($j=0;$j -lt 4;$j++){ [void](Add-Rect $s $x $tableY $colW[$j] 38 $C.Navy); [void](Add-Text $s $headers[$j] $x ($tableY+11) $colW[$j] 18 9.5 $C.White $true 2); $x += $colW[$j] }
    for($i=0;$i -lt 3;$i++){
        $x=$tableX; $y=$tableY+38+$i*54; $fill=if($i%2 -eq 0){$C.Light}else{$C.White}
        for($j=0;$j -lt 4;$j++){ [void](Add-Rect $s $x $y $colW[$j] 54 $fill $false $C.Mid); [void](Add-Text $s $rows[$i][$j] ($x+5) ($y+17) ($colW[$j]-10) 21 10.5 $C.Ink ($j -eq 0) 2); $x += $colW[$j] }
    }
    [void](Add-Text $s 'POPULARITY TỪ FINAL TRAIN' 48 356 491 18 9.5 $C.Blue $true)
    $groups=@(
        @('COLD','7.647','f = 0',$C.Gray,$C.Light),
        @('RARE','6.884','f ≤ 6',$C.Blue,$C.PaleBlue),
        @('MEDIUM','16.940','7–26',$C.Teal,$C.PaleTeal),
        @('POPULAR','6.112','f ≥ 27',$C.Orange,$C.PaleOrange)
    )
    for($i=0;$i -lt 4;$i++){
        $x=48+$i*124
        [void](Add-Rect $s $x 386 112 78 $groups[$i][4] $true)
        [void](Add-Text $s $groups[$i][0] $x 397 112 16 8.5 $groups[$i][3] $true 2)
        [void](Add-Text $s $groups[$i][1] $x 420 112 21 15 $C.Navy $true 2)
        [void](Add-Text $s $groups[$i][2] $x 445 112 14 8.5 $C.Gray $true 2)
    }
    Add-Kpi $s '498.849' 'TRAIN PREFIX-TARGET' 570 132 326 $C.PaleBlue $C.Blue
    Add-Kpi $s '23/23' 'AUDIT NỘI BỘ ĐẠT' 570 246 326 $C.PaleTeal $C.Teal
    Add-Kpi $s '11/11' 'PYTEST ĐẠT' 570 360 326 $C.PaleOrange $C.Orange

    # Slide 17
    $s = $presentation.Slides.Add(17, 12)
    Add-Header $s 17 'Pipeline đã được kiểm tra; bước tiếp theo là huấn luyện' 'Dữ liệu sẵn sàng, nhưng kết luận về mô hình vẫn phải chờ thí nghiệm.' 'BÀN GIAO'
    [void](Add-Rect $s 50 132 394 272 $C.PaleTeal $true)
    [void](Add-Text $s 'ĐÃ HOÀN THÀNH' 72 153 350 22 12 $C.Teal $true 2)
    Add-Bullets $s @('EDA và threshold sensitivity','Temporal split + cold-start 20%','Parquet, mapping và RecBole','23/23 audit • 11/11 pytest') 78 200 330 162 13 $C.Ink $C.Teal
    [void](Add-Rect $s 516 132 394 272 $C.PaleBlue $true)
    [void](Add-Text $s 'BƯỚC TIẾP THEO' 538 153 350 22 12 $C.Blue $true 2)
    Add-Bullets $s @('GRU4Rec • NARM • SASRec × 2 seed','Recall • MRR • NDCG @20','Coverage • Gini • ARP','Norm • cosine • t-SNE • rec_count') 544 200 330 162 13 $C.Ink $C.Blue
    [void](Add-Arrow $s 455 270 505 270 $C.Orange 3)
    [void](Add-Rect $s 195 437 570 38 $C.Navy $true)
    [void](Add-Text $s 'Một split • một mapping • một candidate catalog cho cả ba mô hình' 215 448 530 18 11 $C.White $true 2)

    # Slide 18
    $s = $presentation.Slides.Add(18, 12)
    Add-Background $s $C.Navy
    Add-Tag $s 'KẾT LUẬN' 54 40 126 $C.Teal $C.Navy
    [void](Add-Text $s 'Dữ liệu đã đủ điều kiện cho một thí nghiệm công bằng' 54 94 830 72 31 $C.White $true)
    $conclusions=@(
        @('01','DỮ LIỆU PHÙ HỢP','Long-tail mạnh • category có tín hiệu',$C.Blue,$C.PaleBlue),
        @('02','PIPELINE HỢP LỆ','Temporal split • không leakage • cold rõ ràng',$C.Teal,$C.PaleTeal),
        @('03','SẴN SÀNG HUẤN LUYỆN','Cùng split và mapping cho ba mô hình',$C.Orange,$C.PaleOrange)
    )
    for($i=0;$i -lt 3;$i++){
        $x=54+$i*286
        [void](Add-Rect $s $x 218 260 142 $conclusions[$i][4] $true)
        [void](Add-Text $s $conclusions[$i][0] ($x+18) 237 44 28 19 $conclusions[$i][3] $true)
        [void](Add-Text $s $conclusions[$i][1] ($x+68) 240 172 20 10 $C.Navy $true)
        [void](Add-Text $s $conclusions[$i][2] ($x+18) 292 224 46 11 $C.Ink $false 2)
    }
    [void](Add-Text $s 'Chưa kết luận mô hình nào bias hơn cho đến khi hoàn thành huấn luyện.' 86 404 788 30 14 $C.Mid $true 2)
    [void](Add-Text $s 'Measure first  •  Explain second  •  Mitigate next' 86 455 788 26 15 $C.Teal $true 2)
    [void](Add-Text $s '18 / 18' 826 500 82 18 10 $C.Mid $true 3)

    $presentation.SaveAs($outputPath, 24)

    if (Test-Path $previewPath) {
        Remove-Item -LiteralPath $previewPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $previewPath -Force | Out-Null
    $presentation.Export($previewPath, 'PNG', 1600, 900)
}
catch {
    $slideCount = if ($null -ne $presentation) { $presentation.Slides.Count } else { 0 }
    Write-Error ("PowerPoint build failed after slide count {0}: {1}`n{2}" -f $slideCount, $_.Exception.Message, $_.InvocationInfo.PositionMessage)
}
finally {
    if ($null -ne $presentation) {
        try { $presentation.Close() } catch {}
    }
    if ($null -ne $ppt) {
        try { $ppt.Quit() } catch {}
    }
    if ($null -ne $presentation) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) }
    if ($null -ne $ppt) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output $outputPath
