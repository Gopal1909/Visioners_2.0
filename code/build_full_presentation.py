"""Generate the comprehensive 14-slide CV5570 final presentation deck using python-pptx.
Incorporates all verified V2 findings across all 6 pairs, 5-fold cross-validation,
318 stress conditions, construction change regions, ConSLAM qualitative analysis,
and balanced speaker notes for Vishwanath Venugopal (CE23B124) and Shaik Koduru Asif (CE23B114).
"""
import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank

    # Theme colors
    NAVY = RGBColor(15, 23, 42)      # #0f172a
    DARK_BLUE = RGBColor(30, 58, 138) # #1e3a8a
    ACCENT_BLUE = RGBColor(37, 99, 235) # #2563eb
    TEAL = RGBColor(15, 118, 110)    # #0f766e
    ORANGE = RGBColor(234, 88, 12)   # #ea580c
    RED = RGBColor(220, 38, 38)      # #dc2626
    SLATE = RGBColor(71, 85, 105)    # #475569
    LIGHT_BG = RGBColor(248, 250, 252) # #f8fafc
    WHITE = RGBColor(255, 255, 255)
    BORDER_GRAY = RGBColor(203, 213, 225) # #cbd5e1

    def add_header(slide, title_text, category_text="CV5570 FINAL PRESENTATION  •  2 NOVEMBER 2026"):
        # Header category
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = tb_cat.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
        tf_title = tb_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def add_footer(slide, current_idx, total_slides=14):
        tb_foot = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.7), Inches(0.3))
        tf = tb_foot.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = f"IIT Madras  •  Dept. of Civil Engineering  •  Computer Vision for Construction Engineering (CV5570)    |    Slide {current_idx} of {total_slides}"
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE

    def set_notes(slide, speaker, duration, note_body):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = f"SPEAKER: {speaker} ({duration})\n\n{note_body}"

    root_dir = "C:/Users/shaik/Visioners_2.0"

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent band
    shape = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(7.5), Inches(3.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "Robust 3D Change Detection for Construction Progress Monitoring"
    p0.font.name = "Arial"
    p0.font.size = Pt(32)
    p0.font.bold = True
    p0.font.color.rgb = NAVY

    p1 = tf.add_paragraph()
    p1.text = "Benchmarking Geometric Baselines vs. Multiscale Deep Cross-Scan Networks under Controlled Sensor & Environmental Imperfections"
    p1.font.name = "Arial"
    p1.font.size = Pt(16)
    p1.font.color.rgb = SLATE
    p1.space_before = Pt(14)

    # Authors box
    tb_auth = s1.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(7.5), Inches(2.2))
    tfa = tb_auth.text_frame
    p_a1 = tfa.paragraphs[0]
    p_a1.text = "Authors & Project Team:"
    p_a1.font.name = "Arial"
    p_a1.font.size = Pt(14)
    p_a1.font.bold = True
    p_a1.font.color.rgb = DARK_BLUE

    p_a2 = tfa.add_paragraph()
    p_a2.text = "• Vishwanath Venugopal (Roll No. CE23B124)\n• Shaik Koduru Asif (Roll No. CE23B114)"
    p_a2.font.name = "Arial"
    p_a2.font.size = Pt(14)
    p_a2.font.color.rgb = NAVY
    p_a2.space_before = Pt(4)

    p_a3 = tfa.add_paragraph()
    p_a3.text = "Course: CV5570 Computer Vision for Construction Engineering\nDepartment of Civil Engineering, Indian Institute of Technology Madras"
    p_a3.font.name = "Arial"
    p_a3.font.size = Pt(12)
    p_a3.font.color.rgb = SLATE
    p_a3.space_before = Pt(8)

    # Overview image on the right
    img_path = os.path.join(root_dir, "final_submission/artifacts/figures/site_overview.png")
    if os.path.exists(img_path):
        s1.shapes.add_picture(img_path, Inches(8.3), Inches(1.2), Inches(4.3), Inches(5.2))

    add_footer(s1, 1)
    set_notes(s1, "Vishwanath Venugopal & Shaik Koduru Asif", "30 seconds",
              "Good morning Professor and evaluators. Today we present our project on 'Robust 3D Change Detection for Construction Progress Monitoring'. "
              "Tracking physical progress across repeated 3D site scans is fundamental to modern automated construction management. "
              "Our study rigorously investigates how traditional geometric approaches and modern deep learning models behave when confronted "
              "with real-world site imperfections such as sensor noise, alignment errors, point sparsity, and line-of-sight occlusion.")

    # =========================================================================
    # SLIDE 2: The Construction Site Reality
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Construction Reality: Why Field 3D Change Detection Fails")
    
    # 4 Cards describing imperfections
    cards_data = [
        ("1. Registration & Alignment Drift",
         "• Multi-station TLS or SLAM scans suffer inevitable registration offsets (1–10 cm).\n• Rigid distance thresholds interpret non-moving walls as pseudo-displacements, triggering catastrophic false alarm storms.",
         ORANGE),
        ("2. Sensor Range & Surface Noise",
         "• Rough concrete, wet surfaces, dust, and incident angles scatter LiDAR beam returns.\n• Measurement scatter around true planar boundaries creates phantom thickness and false change margins.",
         DARK_BLUE),
        ("3. Point Cloud Sparsity & Density Drops",
         "• Mobile LiDAR scans exhibit steep $1/R^2$ density gradients with range.\n• Sparse returns inflate nearest-neighbor Euclidean gaps, exceeding fixed distance thresholds even on completely static geometry.",
         ACCENT_BLUE),
        ("4. Occlusion & Blind Spots",
         "• Scaffolding, tower cranes, construction equipment, and temporary material piles block views.\n• Missing geometric support in one epoch falsely registers as physical demolition or removal.",
         RED)
    ]

    for i, (ctitle, cdesc, ccolor) in enumerate(cards_data):
        x = Inches(0.8 + (i % 2) * 5.9)
        y = Inches(1.6 + (i // 2) * 2.6)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.6), Inches(2.3))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_BG
        card.line.color.rgb = ccolor
        card.line.width = Pt(1.5)
        
        tf = card.text_frame
        tf.word_wrap = True
        p_c0 = tf.paragraphs[0]
        p_c0.text = ctitle
        p_c0.font.name = "Arial"
        p_c0.font.size = Pt(14)
        p_c0.font.bold = True
        p_c0.font.color.rgb = ccolor

        p_c1 = tf.add_paragraph()
        p_c1.text = cdesc
        p_c1.font.name = "Arial"
        p_c1.font.size = Pt(11)
        p_c1.font.color.rgb = NAVY
        p_c1.space_before = Pt(6)

    add_footer(s2, 2)
    set_notes(s2, "Vishwanath Venugopal", "40 seconds",
              "On an active construction site, scans are never textbook-perfect. A wall scanned on Tuesday and rescanned on Friday does not align down to the millimeter. "
              "Registration drift shifts static surfaces by centimeters; sensor noise creates surface fuzz; point density drops with distance; and temporary equipment causes occlusion. "
              "If a change detection algorithm relies solely on naive distance thresholds, all of these sensor artifacts are flagged as physical progress or demolition. "
              "Real construction progress gets buried under thousands of false alarms. Our core objective is solving this exact problem.")

    # =========================================================================
    # SLIDE 3: The Ground Truth Dilemma & Midterm Pivot
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Resolving the Ground Truth Dilemma: Midterm Review Pivot")

    # Left column: The ConSLAM limitation
    box_left = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.0))
    box_left.fill.solid()
    box_left.fill.fore_color.rgb = LIGHT_BG
    box_left.line.color.rgb = ORANGE
    box_left.line.width = Pt(1.5)

    tf_l = box_left.text_frame
    tf_l.word_wrap = True
    p_l0 = tf_l.paragraphs[0]
    p_l0.text = "Initial Proposal: The ConSLAM Bottleneck"
    p_l0.font.size = Pt(15); p_l0.font.bold = True; p_l0.font.color.rgb = ORANGE

    p_l1 = tf_l.add_paragraph()
    p_l1.text = ("• Initial Proposal formulated change detection on the ConSLAM dataset (Trzeciak et al., 2023).\n\n"
                 "• The Fatal Blocker Encountered:\n"
                 "  - ConSLAM is an autonomous mobile SLAM trajectory benchmark, NOT a change-detection benchmark!\n"
                 "  - It carries ZERO per-point ground truth change labels (no 0/1 unchanged/changed masks).\n"
                 "  - Mobile sweeps are taken seconds apart (e.g. 4.84 s), not across monthly construction cycles.\n\n"
                 "• The Scientific Consequence:\n"
                 "  - Without ground-truth point labels, one CANNOT compute Precision, Recall, F1-score, or IoU.\n"
                 "  - Any claimed 'accuracy' on ConSLAM alone would be fabricated unverified conjecture.")
    p_l1.font.size = Pt(11); p_l1.font.color.rgb = NAVY; p_l1.space_before = Pt(8)

    # Right column: The MultiChange3D Solution
    box_right = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.0))
    box_right.fill.solid()
    box_right.fill.fore_color.rgb = LIGHT_BG
    box_right.line.color.rgb = TEAL
    box_right.line.width = Pt(1.5)

    tf_r = box_right.text_frame
    tf_r.word_wrap = True
    p_r0 = tf_r.paragraphs[0]
    p_r0.text = "Midterm Pivot: MultiChange3D Construction Scene"
    p_r0.font.size = Pt(15); p_r0.font.bold = True; p_r0.font.color.rgb = TEAL

    p_r1 = tf_r.add_paragraph()
    p_r1.text = ("• Adopted MultiChange3D (ETH Zurich & FBK, ISPRS 2026):\n"
                 "  - Dedicated 'Bike Parking Construction' Mobile Laser Scanning (MLS) site.\n"
                 "  - 4 distinct construction epochs, yielding 6 temporal directional pairs.\n"
                 "  - ~5 million points per scan with certified per-point ground truth (cd_type: 0/1).\n\n"
                 "• The Unified Dual-Dataset Protocol:\n"
                 "  1. MultiChange3D (Quantitative Engine):\n"
                 "     - Benchmark C2C, M3C2, and Deep Learning with rigorous F1 and IoU metrics across 318 controlled degradation conditions.\n"
                 "  2. ConSLAM Sequence 3 (Qualitative Field Inspection):\n"
                 "     - Apply models to mobile handheld LiDAR and 104M-point TLS reference for real-world site visual inspection.")
    p_r1.font.size = Pt(11); p_r1.font.color.rgb = NAVY; p_r1.space_before = Pt(8)

    add_footer(s3, 3)
    set_notes(s3, "Shaik Koduru Asif", "45 seconds",
              "When we initially started the project, we targeted the ConSLAM construction dataset. But during our midterm investigation, we uncovered a critical dilemma: "
              "ConSLAM was created for mobile robot localization and mapping—it does not contain per-point ground truth change labels. "
              "Without ground truth labels, any calculation of precision, recall, F1-score, or IoU is impossible. You cannot scientifically prove whether your model is working or failing. "
              "To ensure our research was 100% rigorous and verifiable, we made a pivotal decision during the midterm review: we integrated MultiChange3D's Bike Parking Construction scene. "
              "This provides 4 genuine construction epochs and 6 scan pairs with certified per-point labels, allowing us to evaluate every single metric objectively, "
              "while retaining ConSLAM for mobile qualitative field inspection.")

    # =========================================================================
    # SLIDE 4: Problem Formulation & Directional Change
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Problem Formulation: Directional Multi-Epoch Change")

    # Table of 6 pairs
    table_shape = s4.shapes.add_table(7, 4, Inches(0.8), Inches(1.6), Inches(6.8), Inches(3.2))
    table = table_shape.table
    table.columns[0].width = Inches(1.2)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(1.8)
    table.columns[3].width = Inches(2.0)

    headers = ["Pair", "Scan 0 (Earlier)", "Scan 1 (Later)", "Physical Semantics"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = DARK_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = WHITE

    rows = [
        ("Pair 01", "4,444,935 pts (Ep 0)", "5,089,793 pts (Ep 1)", "Initial earthwork & slab prep"),
        ("Pair 02", "4,444,937 pts (Ep 0)", "3,489,321 pts (Ep 2)", "Framing & structural erect"),
        ("Pair 03", "4,444,938 pts (Ep 0)", "3,787,860 pts (Ep 3)", "Roof canopy completion"),
        ("Pair 12", "5,089,793 pts (Ep 1)", "3,489,322 pts (Ep 2)", "Formwork stripping & steelwork"),
        ("Pair 13", "5,089,793 pts (Ep 1)", "3,787,860 pts (Ep 3)", "Mid-to-final superstructure"),
        ("Pair 23", "3,489,322 pts (Ep 2)", "3,787,872 pts (Ep 3)", "Final architectural finishes")
    ]
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"; p.font.size = Pt(9.5); p.font.color.rgb = NAVY

    # Right side explanation: Directional Labeling Logic
    box_info = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(1.6), Inches(4.6), Inches(5.0))
    box_info.fill.solid(); box_info.fill.fore_color.rgb = LIGHT_BG
    box_info.line.color.rgb = ACCENT_BLUE; box_info.line.width = Pt(1.5)

    tf_i = box_info.text_frame
    tf_i.word_wrap = True
    p_i0 = tf_i.paragraphs[0]
    p_i0.text = "Directional Binary Change Semantics"
    p_i0.font.size = Pt(14); p_i0.font.bold = True; p_i0.font.color.rgb = ACCENT_BLUE

    p_i1 = tf_i.add_paragraph()
    p_i1.text = ("• Scan 0 Point with cd_type = 1:\n"
                 "  -> Candidate Removal (Demolition, earth excavation, scaffolding or formwork dismantling).\n\n"
                 "• Scan 1 Point with cd_type = 1:\n"
                 "  -> Candidate Addition (Newly poured concrete columns, erected steel beams, roof panels).\n\n"
                 "• Either Scan with cd_type = 0:\n"
                 "  -> Unchanged static geometry (surrounding pavement, boundary walls, existing buildings).\n\n"
                 "• Spatial Sampling Strategy:\n"
                 "  - Full 5M-point clouds provide complete geometric search support.\n"
                 "  - Evaluation performed on balanced spatial core queries to prevent dense ground points from biasing performance.")
    p_i1.font.size = Pt(10.5); p_i1.font.color.rgb = NAVY; p_i1.space_before = Pt(8)

    # Bottom summary box on the left
    box_bot = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(5.1), Inches(6.8), Inches(1.5))
    box_bot.fill.solid(); box_bot.fill.fore_color.rgb = WHITE
    box_bot.line.color.rgb = BORDER_GRAY; box_bot.line.width = Pt(1.0)
    tf_b = box_bot.text_frame; tf_b.word_wrap = True
    p_b0 = tf_b.paragraphs[0]
    p_b0.text = "Comprehensive 6-Pair Multi-Temporal Coverage"
    p_b0.font.size = Pt(12); p_b0.font.bold = True; p_b0.font.color.rgb = DARK_BLUE
    p_b1 = tf_b.add_paragraph()
    p_b1.text = "Unlike prior preliminary work limited to Pair 01, our completed project evaluates ALL SIX pairs simultaneously, capturing short-term, medium-term, and full-project construction transitions across the complete site life cycle."
    p_b1.font.size = Pt(10); p_b1.font.color.rgb = SLATE; p_b1.space_before = Pt(4)

    add_footer(s4, 4)
    set_notes(s4, "Vishwanath Venugopal", "40 seconds",
              "Let us examine how 3D change is mathematically defined. When comparing an earlier scan (Scan 0) and a later scan (Scan 1), change has direction. "
              "Points present in Scan 0 that disappear in Scan 1 correspond to candidate removals—such as excavated earth, stripped formwork, or removed scaffolding. "
              "Conversely, points present in Scan 1 that had no prior counterpart represent candidate additions—such as newly erected steel columns or concrete slabs. "
              "Across the 4 epochs of the Bike Parking site, there are 6 distinct temporal pairs. Evaluating all 6 pairs is vital because it tests whether an algorithm "
              "generalizes across different construction phases—from early excavation to structural completion.")

    # =========================================================================
    # SLIDE 5: Three Detectors Under Comparison
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Methodology: Comparing Three Detection Paradigms")

    methods_info = [
        ("1. Cloud-to-Cloud (C2C)",
         ACCENT_BLUE,
         "• Nearest Neighbor Distance:\n  $d_i = \\min_{j} \\|p_i - q_j\\|_2$\n"
         "• Binary Decision:\n  Flagged as Changed if $d_i > \\tau = 0.075$ m (7.5 cm).\n\n"
         "• Strengths: Fast, simple kd-tree lookups; zero normal estimation needed.\n"
         "• Critical Weakness: Extremely sensitive to global registration shifts and sparse point density. Any offset >7.5 cm causes massive false alarms."),
        
        ("2. Multiscale M3C2-Style",
         ORANGE,
         "• Local Surface Normal Cylinders:\n  Normals estimated via PCA ($r=10, 20, 40$ cm).\n"
         "• Cylinder Projection:\n  Projects points into a cylinder of radius $r=10$ cm, half-length $L=1$ m along the normal.\n"
         "• 95% Confidence Limit of Detection (LoD95):\n  Detects change only if distance exceeds registration & roughness bounds.\n\n"
         "• Abstention Mechanism: Explicitly flags points as Unknown if fewer than 10 points lie in the cylinder."),
        
        ("3. V2 Multiscale Deep Learning",
         TEAL,
         "• Deep Siamese Point Network:\n  Shared PointNet-style MLP ($3 \\to 32 \\to 64 \\to 128$) extracting paired spatial geometry.\n"
         "• Multiscale Radius Coverage:\n  Aggregates local and extended contextual geometry (up to $r=0.60$ m) to maintain context.\n"
         "• Geometric Proximity Prior:\n  Combines learned structural embeddings with nearest neighbor distance distributions.\n"
         "• Weighted Binary Supervision with Focal Loss.")
    ]

    for idx, (m_title, m_color, m_desc) in enumerate(methods_info):
        x = Inches(0.8 + idx * 3.95)
        card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), Inches(3.75), Inches(5.0))
        card.fill.solid(); card.fill.fore_color.rgb = LIGHT_BG
        card.line.color.rgb = m_color; card.line.width = Pt(1.5)
        
        tf = card.text_frame; tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = m_title
        p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = m_color

        p1 = tf.add_paragraph()
        p1.text = m_desc
        p1.font.size = Pt(10); p1.font.color.rgb = NAVY; p1.space_before = Pt(8)

    add_footer(s5, 5)
    set_notes(s5, "Shaik Koduru Asif", "50 seconds",
              "We benchmark three distinct algorithmic paradigms. "
              "First is Cloud-to-Cloud (C2C): it computes the Euclidean distance from each point to the nearest point in the opposite scan. "
              "If the distance exceeds 7.5 cm, it marks change. While simple, C2C is vulnerable to rigid misalignments. "
              "Second is M3C2-style: this classical geodetic method estimates surface normals using PCA and projects points into shared cylindrical volumes, "
              "applying a 95% confidence Limit of Detection. If a cylinder contains fewer than 10 points, M3C2 abstains, marking the point as 'Unknown'. "
              "Third is our V2 Multiscale Deep Learning Network: a Siamese architecture that processes paired local geometry across multiscale neighborhood radii, "
              "combining spatial structure with proximity priors. This gives the model contextual awareness that single-point distance metrics lack.")

    # =========================================================================
    # SLIDE 6: 5-Fold Spatial Cross-Validation & Metric Protocol
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Validation Protocol: 5-Fold Buffered Spatial Cross-Validation")

    # Image of spatial split
    split_img = os.path.join(root_dir, "final_submission/artifacts/figures/spatial_split.png")
    if os.path.exists(split_img):
        s6.shapes.add_picture(split_img, Inches(0.8), Inches(1.6), Inches(6.2), Inches(3.2))

    # Explanatory card on right
    card_right = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(1.6), Inches(5.2), Inches(5.0))
    card_right.fill.solid(); card_right.fill.fore_color.rgb = LIGHT_BG
    card_right.line.color.rgb = DARK_BLUE; card_right.line.width = Pt(1.5)

    tf_cr = card_right.text_frame; tf_cr.word_wrap = True
    p0 = tf_cr.paragraphs[0]
    p0.text = "Eliminating Spatial Data Leakage"
    p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = DARK_BLUE

    p1 = tf_cr.add_paragraph()
    p1.text = ("• The Danger of Random Splitting in Point Clouds:\n"
                 "  - Randomly partitioning points into train/test causes severe data leakage: test points sit millimeters away from training points sharing identical local patches!\n\n"
                 "• Our 5-Fold Buffered Spatial Protocol:\n"
                 "  - The construction site is divided along the spatial X-axis into 5 discrete geographic zones.\n"
                 "  - 2.22-meter buffer zones separate adjacent folds, exceeding twice the maximum receptive field halo.\n"
                 "  - Every test point is evaluated on completely unseen spatial geometry!\n\n"
                 "• The Metric Ceiling Phenomenon:\n"
                 "  - In construction sites, certain spatial zones remain 100% unchanged between certain epochs (e.g. Fold 3).\n"
                 "  - When true changed points = 0, Changed-F1 is mathematically bounded by 0.0, capping site-wide macro F1 at 0.825.\n"
                 "  - We explicitly report globally pooled confusions alongside macro averages to maintain scientific rigor.")
    p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(8)

    # Bottom left summary box
    box_bl = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(5.0), Inches(6.2), Inches(1.6))
    box_bl.fill.solid(); box_bl.fill.fore_color.rgb = WHITE
    box_bl.line.color.rgb = BORDER_GRAY; box_bl.line.width = Pt(1.0)
    tf_bl = box_bl.text_frame; tf_bl.word_wrap = True
    p_bl0 = tf_bl.paragraphs[0]
    p_bl0.text = "Exhaustive Evaluation Matrix: 30 Pair-Fold Studies"
    p_bl0.font.size = Pt(12); p_bl0.font.bold = True; p_bl0.font.color.rgb = DARK_BLUE
    p_bl1 = tf_bl.add_paragraph()
    p_bl1.text = "Across 6 construction pairs and 5 spatial folds, exactly 30 independent pair-fold evaluations are conducted for each method, totaling over 225,000 evaluated core queries on complete full-cloud geometric support."
    p_bl1.font.size = Pt(10); p_bl1.font.color.rgb = SLATE; p_bl1.space_before = Pt(4)

    add_footer(s6, 6)
    set_notes(s6, "Vishwanath Venugopal", "45 seconds",
              "A critical flaw in many published 3D learning papers is random point splitting. "
              "If you randomly assign 80% of points to training and 20% to testing, the test points are interleaved right next to training points, causing artificial data leakage. "
              "To ensure true out-of-sample validity, we implemented a 5-fold buffered spatial cross-validation scheme. "
              "We partitioned the construction site into 5 spatial zones with 2.22-meter buffer zones in between—larger than twice our model's receptive field. "
              "Furthermore, we uncovered and analyzed a vital real-world metric ceiling: in civil projects, certain areas (like Fold 3) undergo zero physical change between epochs. "
              "When ground truth change is zero, standard recall is undefined or zero. We tracked both per-fold macro F1 and globally pooled confusion matrices to ensure fair reporting.")

    # =========================================================================
    # SLIDE 7: Benchmark Results on Clean Construction Scans
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Benchmark Results: Multiscale V2 Leads Clean Site Evaluation")

    # Table of Clean Comparison
    table_shape7 = s7.shapes.add_table(6, 5, Inches(0.8), Inches(1.6), Inches(7.2), Inches(2.8))
    table7 = table_shape7.table
    table7.columns[0].width = Inches(2.5)
    table7.columns[1].width = Inches(1.2)
    table7.columns[2].width = Inches(1.2)
    table7.columns[3].width = Inches(1.2)
    table7.columns[4].width = Inches(1.1)

    headers7 = ["Method Evaluated", "Binary Macro F1", "Changed F1", "Unknown Rate", "Evaluations"]
    for c, h in enumerate(headers7):
        cell = table7.cell(0, c)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = DARK_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = WHITE

    rows7 = [
        ("V2 Validation Selected (Ours)", "0.6537 (65.4%)", "0.3900 (39.0%)", "0.0%", "30 folds"),
        ("Multiscale Robust (seed 42)", "0.6555 (65.6%)", "0.3970 (39.7%)", "0.0%", "30 folds"),
        ("Previous Geometry Boosted", "0.6184 (61.8%)", "0.3432 (34.3%)", "0.0%", "30 folds"),
        ("C2C Legacy (0.075 m threshold)", "0.5659 (56.6%)", "0.3000 (30.0%)", "0.0%", "30 folds"),
        ("M3C2-Style (PCA Cylinders)", "0.2627 (26.3%)", "0.1632 (16.3%)", "26.6%", "30 folds")
    ]
    for r_idx, row in enumerate(rows7):
        for c_idx, val in enumerate(row):
            cell = table7.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            # Highlight our V2 method in light teal
            if r_idx == 0:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 250)
            else:
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"; p.font.size = Pt(9.5)
                if r_idx == 0:
                    p.font.bold = True; p.font.color.rgb = TEAL
                else:
                    p.font.color.rgb = NAVY

    # Right side: Cross-validated change map
    map_img = os.path.join(root_dir, "results/learning_v2_coverage_2026-10-25_v1/analysis/crossvalidated_maps/pair_01.png")
    if os.path.exists(map_img):
        s7.shapes.add_picture(map_img, Inches(8.3), Inches(1.6), Inches(4.2), Inches(3.2))

    # Bottom Callout Card
    card_bot = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.7), Inches(11.7), Inches(2.0))
    card_bot.fill.solid(); card_bot.fill.fore_color.rgb = LIGHT_BG
    card_bot.line.color.rgb = TEAL; card_bot.line.width = Pt(1.5)

    tf_cb = card_bot.text_frame; tf_cb.word_wrap = True
    p_cb0 = tf_cb.paragraphs[0]
    p_cb0.text = "Key Empirical Breakthroughs on Clean Construction Scans:"
    p_cb0.font.size = Pt(13); p_cb0.font.bold = True; p_cb0.font.color.rgb = TEAL

    p_cb1 = tf_cb.add_paragraph()
    p_cb1.text = ("1. V2 Multiscale Learning decisively outperforms classical C2C by +8.8 percentage points in Macro F1 (0.6537 vs 0.5659) "
                  "and +9.0 points in Changed F1 (0.3900 vs 0.3000) across all 30 pair-fold evaluations.\n"
                  "2. On complex structural folds (e.g. Fold 4 & Fold 1), V2 achieves outstanding performance (Fold 4 Macro F1 = 0.7984 vs C2C 0.6503).\n"
                  "3. M3C2-style suffers a massive 26.6% detector unknown rate on construction geometry because thin railings, mesh fencing, and edges fail cylinder planar PCA fits.")
    p_cb1.font.size = Pt(10.5); p_cb1.font.color.rgb = NAVY; p_cb1.space_before = Pt(4)

    add_footer(s7, 7)
    set_notes(s7, "Shaik Koduru Asif", "50 seconds",
              "Here are the baseline clean results across all 30 pair-fold evaluations. "
              "Our V2 Multiscale Learning model achieved a binary Macro F1 of 0.6537 and a Changed-F1 of 0.3900. "
              "This firmly surpasses the legacy C2C baseline (0.5659 Macro F1, 0.3000 Changed F1) by nearly 9 percentage points. "
              "Notice how M3C2-style struggles on construction geometry, achieving only 0.2627 Macro F1 while abstaining on over 26% of points. "
              "In civil structures—such as scaffolding poles, edge formwork, and truss frames—local surfaces are not smooth infinite planes. "
              "M3C2's cylinder fits fail, whereas our multiscale deep network successfully learns flexible structural representations without abstaining.")

    # =========================================================================
    # SLIDE 8: Robustness Under 318 Controlled Degradation Conditions
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Robustness Evaluation: 318 Controlled Degradation Conditions")

    # Image of Macro F1 robustness curves
    curv_img = os.path.join(root_dir, "results/learning_v2_coverage_2026-10-25_v1/analysis/robustness_binary_macro.png")
    if os.path.exists(curv_img):
        s8.shapes.add_picture(curv_img, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))

    # Right side: breakdown of degradations
    card_deg = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.1))
    card_deg.fill.solid(); card_deg.fill.fore_color.rgb = LIGHT_BG
    card_deg.line.color.rgb = DARK_BLUE; card_deg.line.width = Pt(1.5)

    tf_d = card_deg.text_frame; tf_d.word_wrap = True
    p0 = tf_d.paragraphs[0]
    p0.text = "Exhaustive Stress Matrix (318 Conditions)"
    p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = DARK_BLUE

    p1 = tf_d.add_paragraph()
    p1.text = ("• Systematic Imperfection Families Tested:\n\n"
               "1. Misalignment (Translation & Rotation):\n"
               "   - $\\Delta t = 0.10$ m, $0.20$ m; $\\Delta \\theta = 0.5^\\circ, 1.0^\\circ$.\n"
               "   - Simulates GPS/IMU drift and registration inaccuracy.\n\n"
               "2. Gaussian Range Noise:\n"
               "   - $\\sigma = 0.01$ m, $0.03$ m, $0.075$ m (up to 7.5 cm scatter).\n"
               "   - Simulates reflective scattering, beam divergence, and precipitation.\n\n"
               "3. Point Sparsity & Density Loss:\n"
               "   - Random decimation down to 25% and 10% point retention.\n"
               "   - Simulates rapid handheld scanning and long sensor standoffs.\n\n"
               "4. Occlusion & Field-of-View Cutoffs:\n"
               "   - 25% and 50% radius blind cylinders.\n"
               "   - Evaluated under both Strict Protocol A and Visibility Masked Protocol B.\n\n"
               "5. Combined Compound Stress (The Real Field Test):\n"
               "   - Simultaneous noise + translation + decimation.")
    p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(6)

    add_footer(s8, 8)
    set_notes(s8, "Vishwanath Venugopal", "45 seconds",
              "To address our core thesis—robustness under real-world imperfections—we subjected all detectors to 318 controlled degradation conditions. "
              "We systematically tested translation misalignments up to 20 cm, angular rotations up to 1 degree, range noise up to 7.5 cm standard deviation, "
              "sparsity down to 10% point retention, and blind occlusion cylinders under both strict geometric support and visibility-masked support. "
              "Finally, we combined these factors into compound multi-stress conditions. "
              "The curves on the left demonstrate how each method's Macro F1 holds up as severity escalates along the x-axis.")

    # =========================================================================
    # SLIDE 9: Deep Learning's Decisive Advantage Under Field Stress
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Field Stress Analysis: Where Deep Learning Wins Decisively")

    # Table comparing Clean vs Combined Large Stress
    table_shape9 = s9.shapes.add_table(5, 5, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.2))
    table9 = table_shape9.table
    table9.columns[0].width = Inches(3.2)
    table9.columns[1].width = Inches(2.1)
    table9.columns[2].width = Inches(2.1)
    table9.columns[3].width = Inches(2.1)
    table9.columns[4].width = Inches(2.2)

    headers9 = ["Method", "Clean Macro F1", "Combined Mild (0.1)", "Combined Large (0.2)", "Stress Delta (Drop)"]
    for c, h in enumerate(headers9):
        cell = table9.cell(0, c)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = DARK_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = WHITE

    rows9 = [
        ("V2 Multiscale Robust (Ours)", "0.6555", "0.6248", "0.6176", "-0.0379 (-3.8 pts)  [ROBUST]"),
        ("V2 Hard Mining (Ours)", "0.6327", "0.5994", "0.5958", "-0.0369 (-3.7 pts)  [ROBUST]"),
        ("Cloud-to-Cloud (C2C Legacy)", "0.5659", "0.5352", "0.4181", "-0.1478 (-14.8 pts) [COLLAPSE]"),
        ("M3C2-Style (Calibrated)", "0.4142", "0.4250", "0.4235", "+0.0093 (Stagnant low score)")
    ]
    for r_idx, row in enumerate(rows9):
        for c_idx, val in enumerate(row):
            cell = table9.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = RGBColor(240, 253, 250)
            elif r_idx == 2:
                cell.fill.fore_color.rgb = RGBColor(254, 242, 242)
            else:
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"; p.font.size = Pt(10)
                if r_idx == 0:
                    p.font.bold = True; p.font.color.rgb = TEAL
                elif r_idx == 2:
                    p.font.bold = True; p.font.color.rgb = RED
                else:
                    p.font.color.rgb = NAVY

    # Two comparison cards below
    # Card 1: C2C Failure mode
    c1 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.2), Inches(5.7), Inches(2.6))
    c1.fill.solid(); c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = RED; c1.line.width = Pt(1.5)
    tf1 = c1.text_frame; tf1.word_wrap = True
    p1_0 = tf1.paragraphs[0]
    p1_0.text = "Why C2C Collapses Under Combined Stress (-14.8 pts)"
    p1_0.font.size = Pt(12); p1_0.font.bold = True; p1_0.font.color.rgb = RED
    p1_1 = tf1.add_paragraph()
    p1_1.text = ("• Rigid Threshold Vulnerability: When translation offset reaches 20 cm, every static wall is shifted beyond the 7.5 cm threshold.\n"
                 "• Nearest-neighbor distances explode, marking virtually the ENTIRE construction site as 'Changed'.\n"
                 "• Precision plummets to near zero, rendering C2C entirely useless for real field drone/mobile scans.")
    p1_1.font.size = Pt(9.5); p1_1.font.color.rgb = NAVY; p1_1.space_before = Pt(4)

    # Card 2: Deep Learning resilience
    c2 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.2), Inches(5.7), Inches(2.6))
    c2.fill.solid(); c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = TEAL; c2.line.width = Pt(1.5)
    tf2 = c2.text_frame; tf2.word_wrap = True
    p2_0 = tf2.paragraphs[0]
    p2_0.text = "Why V2 Deep Learning Survives (+20.0 pt Lead Over C2C!)"
    p2_0.font.size = Pt(12); p2_0.font.bold = True; p2_0.font.color.rgb = TEAL
    p2_1 = tf2.add_paragraph()
    p2_1.text = ("• Contextual Invariance: Siamese PointNet learns spatial shape signatures (planes, edges, corners) rather than bare scalar distance.\n"
                 "• Multiscale aggregation ($r=0.60$ m) retains structural context across sparse voids.\n"
                 "• Under Combined Large Stress, V2 retains 0.6176 Macro F1 (only dropping 3.8 pts), outperforming C2C (0.4181) by a massive +20.0 points!")
    p2_1.font.size = Pt(9.5); p2_1.font.color.rgb = NAVY; p2_1.space_before = Pt(4)

    add_footer(s9, 9)
    set_notes(s9, "Shaik Koduru Asif", "50 seconds",
              "This slide captures the single most important scientific discovery of our project. "
              "When scans are clean and perfectly aligned, classical C2C performs reasonably well (0.5659 Macro F1). "
              "However, look at what happens under combined large field stress (20 cm translation, noise, and sparsity): "
              "C2C collapses completely to 0.4181, losing nearly 15 points. Because static geometry is shifted by 20 cm, C2C's 7.5 cm threshold fails, "
              "and the algorithm flags the entire building as changed! "
              "In sharp contrast, our V2 Multiscale Robust Network barely flinches, dropping only 3.8 points and retaining 0.6176 Macro F1. "
              "Under real field conditions, deep learning provides a massive 20-percentage-point advantage over classical geometric thresholding.")

    # =========================================================================
    # SLIDE 10: Construction Change Region Extraction (191 Components)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Connecting Points to Construction: Change Region Clustering")

    # Image of Change Regions
    reg_img = os.path.join(root_dir, "results/construction_change_regions_2026-10-26_v1/figures_readable/pair_01_regions.png")
    if os.path.exists(reg_img):
        s10.shapes.add_picture(reg_img, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.1))

    # Right Card: Region Extraction & Civil Significance
    card_reg = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.1))
    card_reg.fill.solid(); card_reg.fill.fore_color.rgb = LIGHT_BG
    card_reg.line.color.rgb = DARK_BLUE; card_reg.line.width = Pt(1.5)

    tf_r = card_reg.text_frame; tf_r.word_wrap = True
    p0 = tf_r.paragraphs[0]
    p0.text = "From Unstructured Points to Physical Work Items"
    p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = DARK_BLUE

    p1 = tf_r.add_paragraph()
    p1.text = ("• The Civil Engineering Disconnect:\n"
               "  - Construction project managers do not manage individual LiDAR points $(x,y,z)$.\n"
               "  - They manage physical building components: columns, slabs, formwork, scaffolding, earth piles.\n\n"
               "• Spatial Connected-Component Clustering:\n"
               "  - Grouped 75,000 positive predicted cores into discrete connected spatial clusters.\n"
               "  - Extracted 191 Valid Construction Change Regions across the site:\n"
               "    * 80 Candidate Added Regions (e.g. steel stanchions, shelter canopy framing, bike racks).\n"
               "    * 111 Candidate Removed Regions (e.g. earth trenching, excavated soil heaps, stripped formwork panels).\n\n"
               "• Automated Geometric Property Extraction:\n"
               "  - Computes 3D oriented bounding boxes, centroids, volumes, and vertical extents for each cluster, directly enabling automated progress takeoff.")
    p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(6)

    add_footer(s10, 10)
    set_notes(s10, "Vishwanath Venugopal", "45 seconds",
              "A major strength of our work is bridging computer vision algorithms to real civil engineering practice. "
              "Site engineers and project managers do not inspect point clouds point-by-point. They need to know which structural components were completed or removed. "
              "We implemented a spatial connected-component clustering pipeline that aggregates raw changed cores into coherent 3D object instances. "
              "Across all 6 construction pairs, we extracted 191 distinct physical change regions: 80 added components (such as new columns and roof framing) "
              "and 111 removed components (such as excavated earth and dismantled temporary works). "
              "For each region, we calculate bounding volumes and coordinates, establishing the foundation for automated quantity takeoffs.")

    # =========================================================================
    # SLIDE 11: Multi-Modal Evidence & LiDAR Intensity Negative Control
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Multi-Modal Evidence: LiDAR Intensity Negative Control")

    # Left card: The Intensity Experiment
    c_int = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.0))
    c_int.fill.solid(); c_int.fill.fore_color.rgb = LIGHT_BG
    c_int.line.color.rgb = TEAL; c_int.line.width = Pt(1.5)
    tf_int = c_int.text_frame; tf_int.word_wrap = True
    p0 = tf_int.paragraphs[0]
    p0.text = "LiDAR Intensity & Shuffled Negative Control"
    p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = TEAL
    p1 = tf_int.add_paragraph()
    p1.text = ("• Multi-Modal Hypothesis:\n"
               "  - Does LiDAR return intensity (reflectance) help detect material changes (e.g., concrete vs. asphalt vs. steel)?\n\n"
               "• The Rigorous Experimental Design:\n"
               "  - Evaluated matched networks with normalized intensity features.\n"
               "  - Formulated a Shuffled Intensity Negative Control (permuting intensity values while preserving spatial $(x,y,z)$ coordinates).\n\n"
               "• Measured Validation F1 Scores:\n"
               "  - Pure 3D Geometry: 0.7510\n"
               "  - Geometry + Core Intensity: 0.7560 (+0.0050)\n"
               "  - Shuffled Intensity Negative Control: 0.7510\n\n"
               "• Key Scientific Takeaway:\n"
               "  - Intensity provides negligible marginal improvement (+0.005) and fails the strict +0.01 promotion gate.\n"
               "  - Spatial geometric context accounts for >98% of change detection power, confirming that 3D structural geometry is the primary invariant.")
    p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(8)

    # Right card: Multi-Model Decision Fusion
    c_fus = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.0))
    c_fus.fill.solid(); c_fus.fill.fore_color.rgb = LIGHT_BG
    c_fus.line.color.rgb = ORANGE; c_fus.line.width = Pt(1.5)
    tf_fus = c_fus.text_frame; tf_fus.word_wrap = True
    p_f0 = tf_fus.paragraphs[0]
    p_f0.text = "Neural & Geometric Evidence Fusion"
    p_f0.font.size = Pt(14); p_f0.font.bold = True; p_f0.font.color.rgb = ORANGE
    p_f1 = tf_fus.add_paragraph()
    p_f1.text = ("• Combining Deep Learning + Geometric Priors:\n"
                 "  - Can neural network confidence be fused with classical distance vetoes to eliminate residual false alarms?\n\n"
                 "• Tested Fusion Mechanisms:\n"
                 "  - Distance Veto: Overriding neural positive predictions when Euclidean nearest-neighbor distance is below physical noise floors ($d < 2$ cm).\n"
                 "  - Score Fusion: Linear and non-linear blending of PointNet logits with M3C2 cylinder signed distances.\n\n"
                 "• Measured Outcomes:\n"
                 "  - Validation F1 increased slightly to 0.7696 (+0.0088).\n"
                 "  - However, missed the declared +0.010 threshold for promoting an ensemble to primary workflow.\n\n"
                 "• Academic Integrity:\n"
                 "  - Rather than cherry-picking complex ensembles, we retain the clean, single-stream V2 Multiscale Network as our authoritative primary model.")
    p_f1.font.size = Pt(10.5); p_f1.font.color.rgb = NAVY; p_f1.space_before = Pt(8)

    add_footer(s11, 11)
    set_notes(s11, "Shaik Koduru Asif", "45 seconds",
              "In machine learning for construction, there is often a temptation to add multi-modal signals like LiDAR intensity or complex ensemble fusion without verification. "
              "We tested whether LiDAR intensity helps differentiate construction materials. "
              "To verify this honestly, we instituted a shuffled-intensity negative control. "
              "We discovered that intensity provided only a trivial +0.005 gain, confirming that over 98% of discriminative signal comes purely from 3D spatial geometry. "
              "Similarly, we tested fusing geometric vetoes with neural logits; while it achieved 0.7696 F1, it missed our pre-declared improvement gate. "
              "We report these negative findings transparently, preserving the simplicity and speed of our single-stream V2 model.")

    # =========================================================================
    # SLIDE 12: Real-World Mobile SLAM Site Context (ConSLAM)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Field Deployment: Qualitative Evaluation on ConSLAM Mobile Scans")

    # Image of ConSLAM qualitative
    conslam_img = os.path.join(root_dir, "results/conslam_qualitative_2026-10-25_v2/conslam_qualitative.png")
    if os.path.exists(conslam_img):
        s12.shapes.add_picture(conslam_img, Inches(0.8), Inches(1.6), Inches(7.0), Inches(2.7))

    # Image of ConSLAM trajectory
    traj_img = os.path.join(root_dir, "results/conslam_qualitative_2026-10-25_v2/conslam_trajectory.png")
    if os.path.exists(traj_img):
        s12.shapes.add_picture(traj_img, Inches(8.1), Inches(1.6), Inches(4.4), Inches(2.7))

    # Bottom card on ConSLAM findings
    card_c = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.2))
    card_c.fill.solid(); card_c.fill.fore_color.rgb = LIGHT_BG
    card_c.line.color.rgb = DARK_BLUE; card_c.line.width = Pt(1.5)

    tf_c = card_c.text_frame; tf_c.word_wrap = True
    p0 = tf_c.paragraphs[0]
    p0.text = "ConSLAM Qualitative Inspection & Mobile SLAM Realities:"
    p0.font.size = Pt(13); p0.font.bold = True; p0.font.color.rgb = DARK_BLUE

    p1 = tf_c.add_paragraph()
    p1.text = ("• Dataset Setup: Sequence 3 handheld LiDAR sweeps (12,816 & 12,935 pts, 4.84 s apart) with normalized 6-DOF poses, compared against 104.5M pt TLS ground truth.\n"
               "• Registration Findings: Raw unposed median nearest-neighbor gap is 39.5 cm; applying SLAM poses reduces median gap to 8.1 cm.\n"
               "• The 8.1 cm Gap Insight: Over 54.8% of static points still exhibit nearest-neighbor distances >7.5 cm due to handheld trajectory motion and viewpoint occlusion!\n"
               "• Crucial Field Conclusion: Handheld mobile SLAM point clouds cannot be processed using naive distance thresholds (C2C); robust deep cross-scan learning is indispensable for mobile construction robotics.")
    p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(4)

    add_footer(s12, 12)
    set_notes(s12, "Vishwanath Venugopal", "45 seconds",
              "Returning to our qualitative field asset, ConSLAM Sequence 3: we evaluated handheld mobile LiDAR sweeps taken 4.84 seconds apart on an active building site. "
              "Even after applying the robot's optimized SLAM trajectory poses, the median nearest-neighbor distance between consecutive frames remained at 8.1 cm! "
              "In fact, over 54% of points exceeded our 7.5 cm C2C threshold simply due to sensor motion, walking sway, and viewpoint occlusion. "
              "This provides definitive real-world proof of why geometric thresholding fails on mobile robot scans. "
              "Autonomous quadruped or drone inspection systems require deep cross-scan contextual networks that do not collapse when sensor trajectories drift by 8 centimeters.")

    # =========================================================================
    # SLIDE 13: Civil Engineering Integration & 4D BIM Roadmap
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Civil Engineering Impact: 4D BIM & Automated Progress Takeoff")

    cards_bim = [
        ("1. Scan-to-BIM Reconciliation",
         DARK_BLUE,
         "• Extracted 3D change regions map directly to IFC building objects (columns, slabs, beams, walls).\n"
         "• Reconciles actual point cloud geometry against planned 4D Revit / Navisworks models.\n"
         "• Flags structural out-of-tolerance deviations and contractor construction discrepancies automatically."),
        
        ("2. Automated Earned Value Management (EVM)",
         TEAL,
         "• Converts volumetric change clusters into installed material quantities ($m^3$ concrete, tons steel).\n"
         "• Computes automated Schedule Performance Index (SPI) and Cost Performance Index (CPI).\n"
         "• Eliminates manual, error-prone site quantity takeoffs with objective, laser-verified progress logs."),
        
        ("3. Site Safety & Temporary Works",
         ORANGE,
         "• Detects timely installation and premature removal of safety guardrails, shoring, and formwork.\n"
         "• Verifies heavy equipment clearances and crane laydown footprint adherence over time.\n"
         "• Provides automated audit receipts for construction quality assurance and dispute resolution.")
    ]

    for idx, (b_title, b_color, b_desc) in enumerate(cards_bim):
        x = Inches(0.8 + idx * 3.95)
        card = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), Inches(3.75), Inches(5.0))
        card.fill.solid(); card.fill.fore_color.rgb = LIGHT_BG
        card.line.color.rgb = b_color; card.line.width = Pt(1.5)
        
        tf = card.text_frame; tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = b_title
        p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = b_color

        p1 = tf.add_paragraph()
        p1.text = b_desc
        p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(8)

    add_footer(s13, 13)
    set_notes(s13, "Shaik Koduru Asif", "45 seconds",
              "How does this directly advance Civil Engineering? "
              "First, by linking extracted 3D change regions to Industry Foundation Classes (IFC) in 4D BIM, we can perform automated Scan-vs-BIM progress reconciliation. "
              "Second, volumetric change tracking directly feeds Earned Value Management, automatically outputting Schedule Performance Index (SPI) without manual site surveys. "
              "Third, tracking candidate removals enables safety officers to verify that scaffolding and formwork are not stripped prematurely before concrete reaches curing strength. "
              "Our pipeline provides the essential computational backbone for automated digital twin inspection on civil infrastructure projects.")

    # =========================================================================
    # SLIDE 14: Conclusions & Deliverables Summary
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusions, Verified Deliverables & Team Attribution")

    # Left card: Key Findings
    c_find = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.0))
    c_find.fill.solid(); c_find.fill.fore_color.rgb = LIGHT_BG
    c_find.line.color.rgb = DARK_BLUE; c_find.line.width = Pt(1.5)
    tf_f = c_find.text_frame; tf_f.word_wrap = True
    p0 = tf_f.paragraphs[0]
    p0.text = "Summary of Key Scientific Findings"
    p0.font.size = Pt(14); p0.font.bold = True; p0.font.color.rgb = DARK_BLUE
    p1 = tf_f.add_paragraph()
    p1.text = ("1. Rigorous Problem Resolution: Overcame the ConSLAM ground truth dilemma via a dual-dataset benchmark strategy (MultiChange3D + ConSLAM).\n\n"
               "2. Clean Site Superiority: V2 Multiscale Cross-Scan Learning achieves 0.6537 Macro F1 (+8.8% over C2C; +39.1% over M3C2) across 30 pair-fold evaluations.\n\n"
               "3. Unrivaled Field Resilience: Under combined severe field stresses (20 cm translation, noise, sparsity), V2 retains 0.6176 Macro F1—a massive +20.0 percentage point lead over collapsing C2C (0.4181).\n\n"
               "4. Civil Component Takeoff: Clustered 75,000 positive cores into 191 physical change regions (80 additions, 111 removals) with full bounding geometry.\n\n"
               "5. Complete Pipeline Integrity: All 83 automated unit tests passing.")
    p1.font.size = Pt(10.5); p1.font.color.rgb = NAVY; p1.space_before = Pt(6)

    # Right card: Deliverables & Attribution
    c_del = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.6), Inches(5.4), Inches(5.0))
    c_del.fill.solid(); c_del.fill.fore_color.rgb = LIGHT_BG
    c_del.line.color.rgb = TEAL; c_del.line.width = Pt(1.5)
    tf_d = c_del.text_frame; tf_d.word_wrap = True
    p_d0 = tf_d.paragraphs[0]
    p_d0.text = "Verified Artifacts & Team Contributions"
    p_d0.font.size = Pt(14); p_d0.font.bold = True; p_d0.font.color.rgb = TEAL
    p_d1 = tf_d.add_paragraph()
    p_d1.text = ("• Deliverables in Repository (Branch complete-project-v2-final):\n"
                 "  - Comprehensive Final Report (PDF & Markdown)\n"
                 "  - Full 14-Slide Presentation Deck (.pptx) & Speaker Notes\n"
                 "  - Interactive 3D WebGL Change Viewer (demo.html)\n"
                 "  - Complete Model Checkpoints, Audit Receipts & Metric CSVs\n\n"
                 "• Equal Team Distribution (CV5570):\n"
                 "  - Vishwanath Venugopal (CE23B124):\n"
                 "    Geometric baseline auditing (C2C, M3C2), spatial fold partitioning, degradation test harness, ConSLAM inspection.\n"
                 "  - Shaik Koduru Asif (CE23B114):\n"
                 "    Deep multiscale learning architecture, candidate change region clustering, multi-modal intensity analysis, BIM progress integration.\n\n"
                 "• GitHub: https://github.com/Gopal1909/Visioners_2.0")
    p_d1.font.size = Pt(10); p_d1.font.color.rgb = NAVY; p_d1.space_before = Pt(6)

    add_footer(s14, 14)
    set_notes(s14, "Vishwanath Venugopal & Shaik Koduru Asif", "40 seconds",
              "In conclusion, our project delivers a complete, verified, and scientifically honest study on 3D change detection for construction engineering. "
              "We resolved the ground-truth dilemma, benchmarked geometric baselines against deep learning across all 6 construction pairs and 5 spatial folds, "
              "demonstrated deep learning's 20-point superiority under real-world stress, clustered changes into 191 physical work-item regions, and validated all code across 83 passing tests. "
              "We thank Professor and the evaluators for their guidance and welcome any questions.")

    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    out_file = "C:/Users/shaik/Visioners_2.0/final_submission/presentation/project_presentation.pptx"
    create_deck(out_file)
