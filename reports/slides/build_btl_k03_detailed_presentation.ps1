$ErrorActionPreference = 'Stop'

$root = (Resolve-Path '.').Path
$slidesDir = Join-Path $root 'reports\slides'
$output = Join-Path $slidesDir 'BTL-K03-CO-MUC-LUC-TUNG-PHAN-25-SLIDES.pptx'
$preview = Join-Path $root 'pytest\pptx_preview_section_checklists'
$totalSlides = 25

$figSession = Join-Path $root 'results\figures\eda_session_length_histogram_ecdf.png'
$figLorenz = Join-Path $root 'results\figures\eda_interaction_lorenz_curve.png'
$figThreshold = Join-Path $root 'results\figures\eda_frequency_threshold_tradeoff.png'
$figCategory = Join-Path $root 'results\figures\eda_category_structure_and_transition.png'
$figTemporal = Join-Path $root 'results\figures\preprocessing_temporal_split.png'

function RGB([int]$r,[int]$g,[int]$b){ $r + 256*$g + 65536*$b }
$C=@{
    Navy=RGB 11 31 51; Navy2=RGB 24 50 80; Blue=RGB 47 107 255;
    Teal=RGB 22 179 163; Orange=RGB 255 159 67; Red=RGB 228 87 86;
    Green=RGB 44 160 101; Ink=RGB 23 43 77; Gray=RGB 87 104 132;
    Mid=RGB 198 208 221; Light=RGB 244 247 251; White=RGB 255 255 255;
    PaleBlue=RGB 232 240 255; PaleTeal=RGB 228 248 245; PaleOrange=RGB 255 244 228;
}

function Rect($s,$x,$y,$w,$h,$fill,$round=$true,$line=$null){
    $t=if($round){5}else{1}; $o=$s.Shapes.AddShape($t,$x,$y,$w,$h)
    $o.Fill.Solid(); $o.Fill.ForeColor.RGB=$fill
    if($null -eq $line){$o.Line.Visible=0}else{$o.Line.Visible=-1;$o.Line.ForeColor.RGB=$line;$o.Line.Weight=1}
    $o
}

function Text($s,[string]$value,$x,$y,$w,$h,$size,$color,$bold=$false,$align=1){
    $o=$s.Shapes.AddTextbox(1,$x,$y,$w,$h); $o.Fill.Visible=0; $o.Line.Visible=0
    $o.TextFrame2.MarginLeft=0; $o.TextFrame2.MarginRight=0; $o.TextFrame2.MarginTop=0; $o.TextFrame2.MarginBottom=0
    $o.TextFrame2.WordWrap=-1; $o.TextFrame2.TextRange.Text=$value
    $o.TextFrame2.TextRange.Font.Name='Aptos'; $o.TextFrame2.TextRange.Font.Size=[single]$size
    $o.TextFrame2.TextRange.Font.Fill.ForeColor.RGB=$color; $o.TextFrame2.TextRange.Font.Bold=if($bold){-1}else{0}
    $o.TextFrame2.TextRange.ParagraphFormat.Alignment=$align
    $o
}

function Background($s,$color){ [void](Rect $s 0 0 960 540 $color $false) }

function Footer($s,[int]$n,[string]$section){
    [void](Text $s (("{0:D2} / {1}" -f $n,$totalSlides)) 828 510 80 17 10.5 $C.Blue $true 3)
}

function Header($s,[int]$n,[string]$title,[string]$subtitle,[string]$section){
    Background $s $C.White; [void](Rect $s 0 0 14 540 $C.Blue $false)
    [void](Text $s $title 50 28 860 48 25.5 $C.Navy $true)
    Footer $s $n $section
}

function Callout($s,[string]$label,[string]$body,$x,$y,$w,$h,$fill,$accent){
    [void](Rect $s $x $y $w $h $fill $true)
    [void](Rect $s $x $y 7 $h $accent $true)
    [void](Text $s $label ($x+18) ($y+11) ($w-34) 17 9 $accent $true)
    [void](Text $s $body ($x+18) ($y+33) ($w-34) ($h-40) 11.5 $C.Ink $false)
}

function Card($s,[string]$index,[string]$title,[string]$body,$x,$y,$w,$h,$fill,$accent){
    [void](Rect $s $x $y $w $h $fill $true)
    [void](Text $s $index ($x+16) ($y+14) 42 26 17 $accent $true)
    [void](Text $s $title ($x+62) ($y+15) ($w-78) 28 13.5 $C.Navy $true)
    [void](Text $s $body ($x+20) ($y+52) ($w-40) ($h-61) 12 $C.Ink $false)
}

function Kpi($s,[string]$value,[string]$label,$x,$y,$w,$fill,$accent){
    [void](Rect $s $x $y $w 84 $fill $true); [void](Rect $s $x $y 7 84 $accent $true)
    [void](Text $s $value ($x+18) ($y+13) ($w-28) 31 22 $C.Navy $true)
    [void](Text $s $label ($x+18) ($y+51) ($w-28) 20 9.5 $C.Gray $true)
}

function PictureFit($s,[string]$path,$x,$y,$w,$h){
    $p=$s.Shapes.AddPicture($path,0,-1,0,0,-1,-1); $scale=[Math]::Min($w/$p.Width,$h/$p.Height)
    $p.Width=$p.Width*$scale; $p.Height=$p.Height*$scale; $p.Left=$x+($w-$p.Width)/2; $p.Top=$y+($h-$p.Height)/2; $p
}

function Arrow($s,$x1,$y1,$x2,$y2,$color,$weight=2.5){
    $l=$s.Shapes.AddLine($x1,$y1,$x2,$y2); $l.Line.ForeColor.RGB=$color; $l.Line.Weight=[single]$weight; $l.Line.EndArrowheadStyle=3; $l
}

function FourCards($s,[array]$items){
    $positions=@(@(50,96),@(490,96),@(50,273),@(490,273))
    $fills=@($C.PaleBlue,$C.PaleTeal,$C.PaleOrange,$C.Light)
    $accents=@($C.Blue,$C.Teal,$C.Orange,$C.Red)
    for($i=0;$i -lt 4;$i++){
        Card $s ("{0:D2}" -f ($i+1)) $items[$i][0] $items[$i][1] $positions[$i][0] $positions[$i][1] 420 157 $fills[$i] $accents[$i]
    }
}

function TextWithFigure($s,[array]$items,[string]$figure,[string]$figureLabel,[string]$callout){
    $fills=@($C.PaleBlue,$C.PaleTeal,$C.PaleOrange,$C.Light)
    $accents=@($C.Blue,$C.Teal,$C.Orange,$C.Red)
    if($figure -eq $figLorenz){$leftW=390;$figX=475;$figY=104;$figW=420;$figH=360;$blockH=82}
    elseif($figure -eq $figThreshold){$leftW=350;$figX=420;$figY=104;$figW=490;$figH=335;$blockH=82}
    else{$leftW=310;$figX=380;$figY=105;$figW=530;$figH=205;$blockH=82}
    for($i=0;$i -lt 4;$i++){
        $y=92+$i*88
        [void](Rect $s 50 $y $leftW $blockH $fills[$i] $true)
        [void](Text $s ("{0:D2}" -f ($i+1)) 64 ($y+15) 34 22 14 $accents[$i] $true)
        [void](Text $s $items[$i] 103 ($y+10) ($leftW-68) ($blockH-18) 11.6 $C.Ink $false)
    }
    [void](Text $s $figureLabel $figX 82 $figW 18 11.5 $C.Blue $true 2)
    [void](PictureFit $s $figure $figX $figY $figW $figH)
    if($figure -ne $figLorenz){
        $callY=if($figure -eq $figThreshold){447}else{332}
        $callH=if($figure -eq $figThreshold){39}else{112}
        [void](Rect $s $figX $callY $figW $callH $C.PaleBlue $true)
        [void](Text $s $callout ($figX+20) ($callY+12) ($figW-40) ($callH-20) 12.2 $C.Navy $true 2)
    }
}

function ChartOnly($s,[string]$figure,[string]$kind){
    if($kind -eq 'wide'){
        [void](PictureFit $s $figure 45 92 870 365)
    }
    elseif($kind -eq 'square'){
        [void](PictureFit $s $figure 120 82 720 410)
    }
    else{
        [void](PictureFit $s $figure 90 86 780 405)
    }
}

$ppt=$null; $pres=$null
try{
    $ppt=New-Object -ComObject PowerPoint.Application; $ppt.Visible=-1; try{$ppt.WindowState=2}catch{}
    $pres=$ppt.Presentations.Add(); $pres.PageSetup.SlideWidth=960; $pres.PageSetup.SlideHeight=540

    # 1 — Cover
    $s=$pres.Slides.Add(1,12); Background $s $C.White; [void](Rect $s 0 0 18 540 $C.Blue $false)
    [void](Rect $s 54 38 260 34 $C.PaleBlue $true); [void](Text $s 'BTL-K03  •  BẢN THUYẾT TRÌNH CHI TIẾT' 67 47 234 18 10.5 $C.Blue $true 2)
    [void](Text $s 'Thiên lệch độ phổ biến và hình học không gian embedding' 54 105 830 102 32 $C.Navy $true)
    [void](Text $s 'trong mô hình Session-Based Recommendation' 55 218 820 38 21 $C.Blue $true)
    [void](Rect $s 55 300 805 116 $C.PaleBlue $true)
    [void](Text $s 'Vấn đề nghiên cứu  →  EDA Diginetica  →  Tiền xử lý  →  Dữ liệu sẵn sàng huấn luyện' 82 335 750 50 16 $C.Navy $true 2)
    [void](Rect $s 55 445 610 44 $C.PaleOrange $true)
    [void](Text $s 'Đã hoàn thành notebook 01–02; chưa huấn luyện ba mô hình.' 75 457 570 22 12.5 $C.Orange $true)
    [void](Text $s '01 / 25' 823 505 85 18 10.5 $C.Blue $true 3)

    # 2 — SBR
    $s=$pres.Slides.Add(2,12); Header $s 2 'Session-Based Recommendation là gì?' 'Trước khi nói về bias, cần hiểu chính xác dữ liệu đi vào mô hình và mục tiêu dự đoán.' 'MỞ ĐẦU'
    FourCards $s @(
        @('Đơn vị phân tích','Một session là chuỗi các item mà người dùng tương tác trong một lần truy cập. Bài toán không cần lịch sử dài hạn của user, vì mô hình tập trung vào ý định trong phiên hiện tại.'),
        @('Mục tiêu dự đoán','Từ lịch sử [A, B, C], mô hình tính điểm cho catalog và dự đoán item tiếp theo D. Kết quả cuối thường là danh sách Top-20 item có điểm cao nhất.'),
        @('Cách tạo mẫu','Session [A, B, C, D] có thể tạo [A]→B, [A,B]→C và [A,B,C]→D. Vì vậy session dài đóng góp nhiều mẫu huấn luyện hơn session ngắn.'),
        @('Ý nghĩa của thứ tự','Nếu click bị đảo thứ tự, mô hình sẽ học sai quan hệ chuyển tiếp. Do đó việc sắp event theo đúng thời gian là điều kiện kỹ thuật bắt buộc, không phải thao tác trang trí dữ liệu.')
    )
    [void](Rect $s 190 435 580 45 $C.Navy $true); [void](Text $s 'Hiểu đơn giản: xem các item trước đó để đoán item kế tiếp trong cùng session.' 210 448 540 20 11 $C.White $true 2)

    # 3 — Bias
    $s=$pres.Slides.Add(3,12); Header $s 3 'Popularity bias là vấn đề gì?' 'Một mô hình dự đoán đúng vẫn có thể phân phối cơ hội hiển thị rất không đồng đều.' 'VẤN ĐỀ'
    FourCards $s @(
        @('Định nghĩa','Popularity bias là xu hướng mô hình ưu tiên những item vốn đã xuất hiện nhiều trong train, trong khi item hiếm hoặc item mới ít có cơ hội xuất hiện trong danh sách gợi ý.'),
        @('Vòng lặp phản hồi','Item đã phổ biến được gợi ý nhiều hơn, nhận thêm tương tác rồi tiếp tục trở nên phổ biến. Nếu không đo exposure, hệ thống có thể khuếch đại mất cân bằng vốn có.'),
        @('Accuracy chưa đủ','Recall, MRR và NDCG cho biết dự đoán có đúng hay không. Coverage, recommendation Gini và ARP cho biết các lượt gợi ý có tập trung vào item phổ biến hay không.'),
        @('Câu hỏi của đề tài','Đề tài so sánh GRU4Rec, NARM và SASRec trên cùng dữ liệu để xem mô hình nào tập trung Top-20 mạnh hơn và hiện tượng đó có đi kèm khác biệt embedding hay không.')
    )
    [void](Rect $s 175 435 610 45 $C.PaleOrange $true); [void](Text $s 'Không đánh đổi mù quáng: Gini thấp nhưng accuracy gần ngẫu nhiên cũng không phải kết quả tốt.' 195 447 570 23 10.5 $C.Red $true 2)

    # 4 — Definitions
    $s=$pres.Slides.Add(4,12); Header $s 4 'Phân biệt popular, rare, warm và cold item' 'Các nhóm này không đồng nghĩa và phải được định nghĩa chỉ bằng dữ liệu train.' 'KHÁI NIỆM'
    FourCards $s @(
        @('Popular item','Item xuất hiện nhiều lần trong final train. Nó có nhiều ví dụ học và thường nhận nhiều lần cập nhật embedding hơn các item ở phần đuôi.'),
        @('Rare item','Item vẫn xuất hiện trong train nhưng có tần suất thấp hơn. Rare không phải cold vì mô hình vẫn nhìn thấy item và vẫn có thể cập nhật embedding của nó.'),
        @('Warm item','Mọi item có ít nhất một interaction trong train đều là warm. Trong pipeline chính, warm item còn phải vượt ngưỡng hỗ trợ tối thiểu sau fixed-point filtering.'),
        @('Cold item','Cold có frequency bằng 0 trong train. Controlled cold được chủ động holdout; natural temporal cold chỉ xuất hiện sau mốc train. Hai loại được theo dõi riêng.')
    )
    [void](Rect $s 190 435 580 45 $C.Navy $true); [void](Text $s 'Trục tần suất: popular → medium → rare (>0) → cold (=0 trong train).' 210 448 540 20 11 $C.White $true 2)

    # 5 — Research design
    $s=$pres.Slides.Add(5,12); Header $s 5 'Ý tưởng nghiên cứu: nối đầu ra với embedding' 'Đề tài là nghiên cứu chẩn đoán, chưa phải một thuật toán khử bias mới.' 'PHƯƠNG PHÁP'
    FourCards $s @(
        @('So sánh ba mô hình','GRU4Rec, NARM và SASRec phải dùng cùng split, mapping, candidate catalog, nhóm popularity và Top-20. Chạy hai seed giúp kiểm tra kết luận có ổn định hay không.'),
        @('Đo chất lượng','Recall@20, MRR@20 và NDCG@20 là sanity check để chứng minh mô hình học được tín hiệu next-item, thay vì tạo danh sách đa dạng nhưng gần ngẫu nhiên.'),
        @('Đo output bias','Từ tất cả Top-20, đếm recommendation count theo item rồi tính Coverage@20, recommendation Gini và ARP. Đây mới là các chỉ số bias của đầu ra mô hình.'),
        @('Phân tích embedding','Tính norm, Spearman với train frequency, cosine cùng/khác category và t-SNE. Sau đó liên hệ embedding norm với recommendation count ở cấp item.')
    )
    [void](Rect $s 155 435 650 45 $C.PaleTeal $true); [void](Text $s 'Trạng thái hiện tại: đã chuẩn bị dữ liệu; các kết quả mô hình và embedding vẫn là bước tiếp theo.' 175 447 610 23 10.5 $C.Teal $true 2)

    # 6 — Dataset
    $s=$pres.Slides.Add(6,12); Header $s 6 'Diginetica cung cấp đúng loại dữ liệu đề tài cần' 'Dataset có session, thời gian, long-tail và category; quy mô vẫn khả thi cho bài tập lớn.' 'DỮ LIỆU'
    Kpi $s '1.235.380' 'INTERACTION' 50 126 198 $C.PaleBlue $C.Blue
    Kpi $s '310.324' 'SESSION' 264 126 198 $C.PaleTeal $C.Teal
    Kpi $s '122.993' 'ITEM CÓ TƯƠNG TÁC' 478 126 198 $C.PaleOrange $C.Orange
    Kpi $s '1.217' 'CATEGORY METADATA' 692 126 204 $C.Light $C.Red
    Card $s '01' 'train-item-views.csv' 'Chứa sessionId, itemId, eventdate và timeframe. Đây là nguồn chính để sắp chuỗi, chia theo thời gian và tạo mẫu next-item.' 50 235 270 160 $C.Light $C.Blue
    Card $s '02' 'product-categories.csv' 'Ánh xạ item sang category. File này phục vụ kiểm tra category transition, phân tầng controlled cold và cosine embedding sau huấn luyện.' 345 235 270 160 $C.Light $C.Teal
    Card $s '03' 'products.csv' 'Dùng để kiểm tra catalog và tính toàn vẹn metadata. Các trường giá/tên chưa được dùng vì ba baseline hiện tại là mô hình item-ID-based.' 640 235 256 160 $C.Light $C.Orange
    [void](Rect $s 142 425 676 50 $C.Navy $true); [void](Text $s '152 ngày có quan sát trong khoảng 01/01–01/06/2016 • ba file liên kết bằng itemId' 162 439 636 23 10.5 $C.White $true 2)

    # 7 — EDA checklist
    $s=$pres.Slides.Add(7,12); Header $s 7 'Phần 1 — EDA: chúng ta đã kiểm tra những gì?' '' 'EDA'
    Card $s '01' 'Chất lượng và liên kết dữ liệu' 'Kiểm tra schema, missing, duplicate, khóa itemId, ngày và khả năng liên kết ba file raw.' 50 95 420 105 $C.PaleBlue $C.Blue
    Card $s '02' 'Thứ tự và thời gian session' 'Kiểm tra thứ tự event, timeframe trùng nhau, session qua nửa đêm và khoảng ngày quan sát.' 490 95 420 105 $C.PaleTeal $C.Teal
    Card $s '03' 'Đặc điểm session' 'Phân tích độ dài session, session chỉ có một event, click lặp và ảnh hưởng của session dài.' 50 215 420 105 $C.PaleOrange $C.Orange
    Card $s '04' 'Long-tail và popularity đầu vào' 'Đo tần suất item, tỷ trọng top item, đường Lorenz và Gini của interaction raw.' 490 215 420 105 $C.Light $C.Red
    Card $s '05' 'Category và tuổi item' 'Kiểm tra chuyển tiếp cùng category và mối liên hệ giữa thời gian xuất hiện với frequency.' 50 335 420 105 $C.PaleBlue $C.Blue
    Card $s '06' 'Độ nhạy ngưỡng lọc' 'So sánh các ngưỡng frequency để thấy số event, session, item và long-tail bị thay đổi ra sao.' 490 335 420 105 $C.PaleTeal $C.Teal
    [void](Rect $s 195 458 570 36 $C.PaleBlue $true); [void](Text $s 'Các slide tiếp theo lần lượt trình bày kết quả của sáu nhóm kiểm tra này.' 215 467 530 20 12 $C.Navy $true 2)

    # 8 — Data quality
    $s=$pres.Slides.Add(8,12); Header $s 8 'EDA xác nhận dữ liệu đủ sạch để tiếp tục' 'Notebook 01 kiểm tra phiên bản file, schema, missing, duplicate, ngày và liên kết metadata.' 'EDA'
    FourCards $s @(
        @('Liên kết đầy đủ','Không item nào đã tương tác bị thiếu category hoặc product metadata. Ba SHA-256 được lưu lại để notebook 02 xác nhận đang dùng đúng phiên bản raw đã EDA.'),
        @('Không có lỗi khóa chính','Không có dòng trùng hoàn toàn, không có event trùng khóa, không có itemId trùng trong hai bảng metadata và không có timeframe âm.'),
        @('Ngày và catalog hợp lệ','Tất cả eventdate đều parse được. Có 61.054 item trong metadata chưa từng xuất hiện trong interaction; đây là catalog chưa hoạt động, không phải lỗi liên kết.'),
        @('UserId không dùng','UserId thiếu 69,81%, nhưng không cản trở SBR vì đơn vị học là session. Pipeline không tự ý điền userId và không mở rộng sang mô hình hóa người dùng.')
    )
    [void](Rect $s 190 435 580 45 $C.PaleTeal $true); [void](Text $s 'Kết luận EDA chất lượng: dữ liệu đủ tin cậy, nhưng vẫn phải sửa thứ tự click.' 210 448 540 20 11 $C.Teal $true 2)

    # 9 — Ordering/time
    $s=$pres.Slides.Add(9,12); Header $s 9 'Vấn đề nghiêm trọng nhất là thứ tự event trong session' 'Nếu chuỗi sai, mọi mô hình tuần tự đều học quan hệ chuyển tiếp sai.' 'EDA'
    FourCards $s @(
        @('58,21% session sai thứ tự raw','Có 180.643 session không tăng theo timeframe nếu giữ nguyên thứ tự dòng trong CSV. Vì vậy không thể dùng row order làm thứ tự chuỗi chính.'),
        @('Quy tắc sắp ổn định','Event được sắp theo sessionId → timeframe → row_order. Timeframe xác định thứ tự trong session; row_order chỉ dùng để phá hòa khi thời gian bằng nhau.'),
        @('Bốn nhóm trùng timeframe','Các nhóm trùng gồm hai item khác nhau tại cùng một thời điểm. Không có căn cứ để xóa một event, nên pipeline giữ cả hai và dùng stable sort.'),
        @('260 session qua nửa đêm','Các session này kéo dài tối đa khoảng 19 phút 57 giây và chỉ đi qua hai ngày liên tiếp. Pipeline giữ nguyên session và chia tập theo ngày event cuối.')
    )
    [void](Rect $s 180 435 600 45 $C.Navy $true); [void](Text $s 'Quyết định: sửa thứ tự bên trong session, nhưng không chia nhỏ session theo ngày.' 200 448 560 20 11 $C.White $true 2)

    # 10 — Session chart
    $s=$pres.Slides.Add(10,12); Header $s 10 'Biểu đồ 1: phân bố độ dài session' '' 'EDA'; ChartOnly $s $figSession 'wide'

    # 11 — Session explanation
    $s=$pres.Slides.Add(11,12); Header $s 11 'Biểu đồ độ dài session cho biết điều gì?' '' 'EDA'
    FourCards $s @(
        @('Cách đọc biểu đồ','Biểu đồ cột đếm số session theo độ dài. Đường ECDF cho biết tỷ lệ session có độ dài nhỏ hơn hoặc bằng một giá trị trên trục ngang.'),
        @('Kết quả chính','Median session length bằng 3 và 77,07% session dài không quá 5. Diginetica chủ yếu gồm các chuỗi hành vi ngắn.'),
        @('Quyết định xử lý','29,23% session chỉ có một event nên không tạo được lịch sử → target và phải loại. Click lặp được giữ vì có thể là hành vi thật.'),
        @('Điểm cần kiểm soát','Session dài hơn 10 chỉ chiếm 6,59% nhưng chứa 24,72% event. All-prefix training khiến chúng đóng góp nhiều mẫu hơn session ngắn.')
    )
    [void](Rect $s 170 445 620 42 $C.PaleBlue $true); [void](Text $s 'Kết luận: bỏ session length 1, giữ click lặp và theo dõi ảnh hưởng của session dài.' 190 456 580 20 12 $C.Navy $true 2)

    # 12 — Lorenz chart
    $s=$pres.Slides.Add(12,12); Header $s 12 'Biểu đồ 2: đường Lorenz của interaction theo item' '' 'EDA'; ChartOnly $s $figLorenz 'square'

    # 13 — Lorenz explanation
    $s=$pres.Slides.Add(13,12); Header $s 13 'Đường Lorenz chứng minh long-tail như thế nào?' '' 'EDA'
    FourCards $s @(
        @('Cách đọc biểu đồ','Trục ngang là tỷ lệ item tích lũy; trục dọc là tỷ lệ interaction tích lũy. Đường đỏ càng xa đường chéo thì phân phối càng bất bình đẳng.'),
        @('Độ dài phần đuôi','34,20% item chỉ xuất hiện đúng một lần và 63,80% item xuất hiện dưới 5 lần. Catalog có phần đuôi rất dài và thưa.'),
        @('Mức độ tập trung','Top 1% item nhận 21,65% interaction; top 20% item nhận 77,54%. Gini đầu vào bằng 0,726.'),
        @('Giới hạn kết luận','Kết quả này chứng minh dữ liệu có long-tail, chưa chứng minh mô hình bị bias. Recommendation Gini chỉ được tính sau khi có Top-20 của mô hình.')
    )
    [void](Rect $s 185 445 590 42 $C.PaleOrange $true); [void](Text $s 'Diginetica phù hợp để nghiên cứu popularity bias, nhưng bias đầu ra vẫn phải đo sau huấn luyện.' 205 455 550 22 12 $C.Red $true 2)

    # 14 — Category chart
    $s=$pres.Slides.Add(14,12); Header $s 14 'Biểu đồ 3: cấu trúc category và chuyển tiếp trong session' '' 'EDA'; ChartOnly $s $figCategory 'wide'

    # 15 — Category explanation
    $s=$pres.Slides.Add(15,12); Header $s 15 'Biểu đồ category có ý nghĩa gì với embedding?' '' 'EDA'
    FourCards $s @(
        @('Ba phần của biểu đồ','Ba ô lần lượt mô tả quy mô category, số category trong mỗi session và tỷ lệ chuyển tiếp cùng category so với baseline ngẫu nhiên.'),
        @('Tín hiệu trong session','80,95% session chỉ chứa một category. Sau khi bỏ self-repeat, 87,45% chuyển tiếp liên tiếp vẫn cùng category.'),
        @('So với ngẫu nhiên','Baseline ngẫu nhiên chỉ đạt 1,08%. Category vì vậy phản ánh cấu trúc hành vi thực, không chỉ là nhãn metadata.'),
        @('Dùng ở bước embedding','Sau huấn luyện sẽ so sánh cosine của cặp cùng/khác category. Phải cân bằng category để category lớn không thống trị kết quả.')
    )
    [void](Rect $s 180 445 600 42 $C.PaleTeal $true); [void](Text $s 'Tuổi item và frequency có Spearman 0,504 nên quan hệ frequency–norm không được diễn giải là nhân quả.' 200 455 560 22 11.8 $C.Teal $true 2)

    # 16 — Threshold chart
    $s=$pres.Slides.Add(16,12); Header $s 16 'Biểu đồ 4: độ nhạy theo minimum item frequency' '' 'EDA'; ChartOnly $s $figThreshold 'normal'

    # 17 — Threshold explanation
    $s=$pres.Slides.Add(17,12); Header $s 17 'Ngưỡng lọc tạo ra sự đánh đổi như thế nào?' '' 'EDA'
    FourCards $s @(
        @('Cách đọc biểu đồ','Bốn ô lần lượt cho biết tỷ lệ event/session còn lại, kích thước item vocabulary, Gini sau lọc và tỷ lệ loại rare item khi tăng ngưỡng.'),
        @('Ngưỡng 2 và 3','Ngưỡng 2 giữ 78.989 item và 88,92% event. Ngưỡng 3 giữ 60.454 item và 85,64% event.'),
        @('Ngưỡng 5','Ngưỡng 5 giữ 42.171 item và 80,07% event, nhưng chỉ còn 34,29% catalog raw. Đây là mức cắt long-tail đáng kể.'),
        @('Cách kết luận đúng','Gini giảm chủ yếu vì item tail bị loại, chưa phải giảm model bias. Ngưỡng 5 là baseline; ngưỡng 2/3 dành cho thí nghiệm độ nhạy.')
    )
    [void](Rect $s 180 445 600 42 $C.PaleOrange $true); [void](Text $s 'Không tuyên bố ngưỡng 5 tối ưu; đây là baseline có giải thích và có số liệu attrition.' 200 455 560 22 12 $C.Red $true 2)

    # 18 — Preprocessing checklist
    $s=$pres.Slides.Add(18,12); Header $s 18 'Phần 2 — Tiền xử lý dữ liệu: đã làm những gì?' '' 'GIẢI PHÁP'
    FourCards $s @(
        @('Khôi phục chuỗi','Sai thứ tự raw được xử lý bằng stable sort. Session length 1 bị loại vì không có target; click lặp được giữ vì có thể là tín hiệu hành vi thật.'),
        @('Kiểm soát thời gian','Session qua nửa đêm được giữ nguyên. Toàn bộ session được gán split theo ngày event cuối để tránh cắt session và hạn chế temporal leakage.'),
        @('Kiểm soát sparsity','Minimum frequency bằng 5 được áp dụng lặp với minimum session length bằng 2 cho tới fixed point, vì xóa item và session tác động qua lại.'),
        @('Kiểm soát leakage','Popularity group, item frequency và catalog có thể học đều được xác định từ final train. Validation/test không được dùng để đặt nhãn popularity.')
    )
    [void](Rect $s 165 445 630 42 $C.PaleBlue $true); [void](Text $s 'Một pipeline chung được dùng cho GRU4Rec, NARM và SASRec để so sánh công bằng.' 185 456 590 20 12 $C.Navy $true 2)

    # 19 — Temporal chart
    $s=$pres.Slides.Add(19,12); Header $s 19 'Biểu đồ 5: số event theo ngày và ranh giới temporal split' '' 'PREPROCESSING'; ChartOnly $s $figTemporal 'wide'

    # 20 — Temporal explanation
    $s=$pres.Slides.Add(20,12); Header $s 20 'Biểu đồ temporal split cần được hiểu như thế nào?' '' 'PREPROCESSING'
    FourCards $s @(
        @('Cách đọc biểu đồ','Đường xanh biểu diễn số event theo ngày. Hai đường dọc đánh dấu thời điểm bắt đầu validation và test.'),
        @('Các mốc thời gian','Train dùng session kết thúc từ 01/01–18/05/2016; validation là 19/05–25/05; test là 26/05–01/06.'),
        @('Tại sao chia theo session?','Gán split theo ngày của event cuối giúp giữ nguyên chuỗi và tránh một session xuất hiện ở hai tập. Ba tập không trùng session.'),
        @('Tại sao dùng 7 ngày?','Mỗi cửa sổ có đủ các thứ trong tuần và vẫn còn hàng chục nghìn mẫu. Đây là baseline thiết kế, chưa được chứng minh tối ưu.')
    )
    [void](Rect $s 150 445 660 42 $C.PaleBlue $true); [void](Text $s 'Nguyên tắc quan trọng nhất: học từ quá khứ và không dùng tương lai để định nghĩa popularity.' 170 455 620 22 12 $C.Navy $true 2)

    # 21 — Fixed point
    $s=$pres.Slides.Add(21,12); Header $s 21 'Fixed-point filtering: tại sao phải lọc lặp?' '' 'PREPROCESSING'
    FourCards $s @(
        @('Hai điều kiện','Mỗi session phải có ít nhất 2 event để tạo target. Mỗi item phải xuất hiện ít nhất 5 lần trong train để có tín hiệu học tối thiểu.'),
        @('Tác động dây chuyền','Nếu item X bị xóa khỏi session [A,X], session chỉ còn [A] và phải xóa. Việc xóa session lại làm frequency của A và các item khác giảm.'),
        @('Trước cold holdout','Train giảm từ 990.708 xuống 841.402 event; từ 189.774 xuống 174.691 session; catalog giảm từ 114.768 xuống 38.237 item.'),
        @('Ý nghĩa nghiên cứu','Attrition được lưu theo từng vòng để minh bạch mức cắt long-tail. Cùng một fixed-point rule phải được dùng cho cả ba mô hình.')
    )
    [void](Rect $s 160 445 640 42 $C.PaleOrange $true); [void](Text $s 'Fixed point nghĩa là tiếp tục lọc cho tới khi số item và session không còn thay đổi.' 180 456 600 20 12 $C.Red $true 2)

    # 22 — Cold
    $s=$pres.Slides.Add(22,12); Header $s 22 'Controlled cold-start 20% được tạo như thế nào?' '' 'PREPROCESSING'
    FourCards $s @(
        @('Chọn controlled cold','Từ 38.237 item train sau lọc, chọn 7.647 item, tương đương 19,99895%. Chọn ở cấp item, phân tầng theo category và dùng seed 2024.'),
        @('Loại khỏi train nhưng giữ mapping','Toàn bộ interaction của controlled cold bị xóa khỏi train. Item ID và category vẫn được giữ để còn một vector embedding khởi tạo cho phân tích minh họa.'),
        @('Giữ đối chứng category','Controlled cold trải trên 629 category và cả 629 category vẫn còn ít nhất một warm peer. Điều này cho phép so sánh cold với warm cùng category.'),
        @('Natural cold là nhóm khác','Có 6.700 item chỉ xuất hiện sau mốc train. Đây là natural temporal cold, được audit riêng và không trộn với controlled cold trong diễn giải.')
    )
    [void](Rect $s 80 445 800 42 $C.PaleBlue $true); [void](Text $s 'Khi train: cold không là positive, không negative sampling, không nhận gradient; phải lưu embedding khởi tạo.' 100 455 760 22 11.8 $C.Navy $true 2)

    # 23 — Final data
    $s=$pres.Slides.Add(23,12); Header $s 23 'Dữ liệu cuối cùng và các nhóm popularity' '' 'KẾT QUẢ'
    $x0=48;$y0=128;$cw=@(162,105,105,105,132);$head=@('TẬP','EVENT','SESSION','ITEM','PREFIX-TARGET')
    $rows=@(@('Train','649.632','150.783','29.936','498.849'),@('Validation warm','39.450','9.622','12.227','29.828'),@('Test warm','52.792','12.791','13.847','40.001'))
    $x=$x0; for($j=0;$j -lt 5;$j++){[void](Rect $s $x $y0 $cw[$j] 36 $C.Navy $false);[void](Text $s $head[$j] $x ($y0+10) $cw[$j] 18 8.7 $C.White $true 2);$x+=$cw[$j]}
    for($i=0;$i -lt 3;$i++){ $x=$x0;$y=$y0+36+$i*50;$f=if($i%2 -eq 0){$C.Light}else{$C.White}; for($j=0;$j -lt 5;$j++){[void](Rect $s $x $y $cw[$j] 50 $f $false $C.Mid);[void](Text $s $rows[$i][$j] ($x+3) ($y+15) ($cw[$j]-6) 20 9.3 $C.Ink ($j -eq 0) 2);$x+=$cw[$j]} }
    [void](Text $s 'NHÓM POPULARITY TỪ FINAL TRAIN' 48 335 520 18 9.5 $C.Blue $true)
    $groups=@(@('COLD','7.647 item','f = 0',$C.Light,$C.Gray),@('RARE','6.884 item','f ≤ 6',$C.PaleBlue,$C.Blue),@('MEDIUM','16.940 item','f = 7–26',$C.PaleTeal,$C.Teal),@('POPULAR','6.112 item','f ≥ 27',$C.PaleOrange,$C.Orange))
    for($i=0;$i -lt 4;$i++){ $x=48+$i*164; [void](Rect $s $x 366 150 91 $groups[$i][3] $true);[void](Text $s $groups[$i][0] $x 378 150 16 8.5 $groups[$i][4] $true 2);[void](Text $s $groups[$i][1] $x 402 150 20 13 $C.Navy $true 2);[void](Text $s $groups[$i][2] $x 431 150 15 8.5 $C.Gray $true 2) }
    [void](Rect $s 720 128 176 329 $C.Navy $true)
    [void](Text $s 'CÁCH HIỂU' 738 150 140 18 9.5 $C.Teal $true 2)
    [void](Text $s "Rare/medium/popular là các warm item đã được học.`n`nNhóm không đúng chính xác 20% vì pipeline giữ các item có cùng frequency trong cùng một nhóm.`n`nMain accuracy chỉ đo trên warm target để phù hợp ba baseline item-ID-based." 740 191 136 224 10.3 $C.White $false)

    # 24 — Outputs/audit
    $s=$pres.Slides.Add(24,12); Header $s 24 'Notebook 02 đã xuất gì và kiểm tra những gì?' '' 'BÀN GIAO'
    FourCards $s @(
        @('Parquet và examples','Xuất event, session sequence, all-prefix examples và last-item examples cho train/validation/test. Bản validation/test full được giữ để audit cold.'),
        @('Mapping và nhóm item','Xuất item mapping, category mapping, popularity groups và cold item list. ID nội bộ bắt đầu từ 1; ID 0 được dành cho padding.'),
        @('Định dạng RecBole','Các file .train.inter, .valid.inter, .test.inter và .item là cùng Diginetica đã xử lý, không phải dataset khác. RecBole không được tự chia ngẫu nhiên lại.'),
        @('Kiểm tra tự động','23/23 audit và 11/11 pytest đạt: không trùng session, không cold trong train, mapping duy nhất, category đầy đủ và popularity chỉ từ train.')
    )
    [void](Rect $s 160 435 640 45 $C.PaleTeal $true); [void](Text $s 'Các kiểm tra chứng minh implementation đúng giao thức; chúng không chứng minh hyperparameter là tối ưu.' 180 447 600 23 10.5 $C.Teal $true 2)

    # 25 — Conclusion
    $s=$pres.Slides.Add(25,12); Background $s $C.White; [void](Rect $s 0 0 14 540 $C.Blue $false)
    [void](Text $s 'Hai notebook đầu đã tạo nền tảng hợp lệ cho thí nghiệm' 54 42 840 52 28 $C.Navy $true)
    Card $s '01' 'Điều đã chứng minh' 'Diginetica có long-tail mạnh; category mang tín hiệu hành vi; raw event cần sắp lại; ngưỡng lọc có trade-off lớn và dữ liệu phải chia theo thời gian.' 54 190 402 160 $C.PaleBlue $C.Blue
    Card $s '02' 'Điều đã hoàn thành' 'Đã tạo temporal split, fixed-point filter, controlled cold 20%, final warm evaluation, mapping, popularity groups, examples và RecBole; toàn bộ audit đều đạt.' 504 190 402 160 $C.PaleTeal $C.Teal
    [void](Rect $s 54 370 852 83 $C.PaleOrange $true)
    [void](Text $s 'CHƯA THỂ KẾT LUẬN' 74 387 190 20 11.5 $C.Orange $true)
    [void](Text $s 'Chưa biết mô hình nào bias hơn, chưa có metric Top-20 và chưa có embedding đã huấn luyện. Bước tiếp theo là train GRU4Rec, NARM, SASRec với hai seed rồi mới liên hệ frequency – norm – recommendation count.' 274 382 608 57 12.2 $C.Navy $false)
    [void](Text $s 'Measure first  •  Explain second  •  Mitigate next' 174 474 612 24 15 $C.Teal $true 2)
    [void](Text $s '25 / 25' 823 505 85 18 10.5 $C.Blue $true 3)

    $pres.SaveAs($output,24)
    if(Test-Path $preview){Remove-Item -LiteralPath $preview -Recurse -Force}
    New-Item -ItemType Directory -Path $preview -Force | Out-Null
    $pres.Export($preview,'PNG',1600,900)
}
catch{
    $count=if($null -ne $pres){$pres.Slides.Count}else{0}
    Write-Error ("Detailed deck failed after slide count {0}: {1}`n{2}" -f $count,$_.Exception.Message,$_.InvocationInfo.PositionMessage)
}
finally{
    if($null -ne $pres){try{$pres.Close()}catch{}}
    if($null -ne $ppt){try{$ppt.Quit()}catch{}}
    if($null -ne $pres){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($pres)}
    if($null -ne $ppt){[void][Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
}
Write-Output $output
