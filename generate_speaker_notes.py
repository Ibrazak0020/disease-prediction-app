# generate_speaker_notes.py
# ---------------------------------------------------
# Detailed Speaker Notes + Architecture Diagram
# for AI Disease Prediction Project
# ---------------------------------------------------

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF

# ---------------------------------------------------
# 1️⃣ Setup Document
# ---------------------------------------------------
pdf_filename = "AI_Disease_Prediction_Speaker_Notes.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=A4, rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", alignment=TA_CENTER, fontSize=18, leading=24, spaceAfter=20))
styles.add(ParagraphStyle(name="Heading", alignment=TA_JUSTIFY, fontSize=14, leading=20, spaceAfter=10, spaceBefore=10))
styles.add(ParagraphStyle(name="Body", alignment=TA_JUSTIFY, fontSize=11, leading=16))

story = []

# ---------------------------------------------------
# 2️⃣ Title Page
# ---------------------------------------------------
story.append(Paragraph("AI DISEASE PREDICTION SYSTEM", styles["TitleCenter"]))
story.append(Paragraph("<b>Student:</b> Kazeem Ibrahim Opeyemi", styles["Body"]))
story.append(Paragraph("<b>Supervisor:</b> Dr. Jet", styles["Body"]))
story.append(Paragraph("<b>Department:</b> Computer Science, Abiola Ajimobi Technical University", styles["Body"]))
story.append(Paragraph("<b>Academic Year:</b> 2024 / 2025", styles["Body"]))
story.append(Spacer(1, 24))
story.append(Paragraph("Speaker / Defense Notes", styles["Heading"]))
story.append(Paragraph("This document provides a detailed explanation of the AI Disease Prediction System, including architecture, methodology, and technical implementation to support project defense.", styles["Body"]))
story.append(PageBreak())

# ---------------------------------------------------
# 3️⃣ Project Overview and Objectives
# ---------------------------------------------------
story.append(Paragraph("1. Project Overview", styles["Heading"]))
story.append(Paragraph("""
The AI Disease Prediction System is a machine learning-based web application that predicts the most probable disease based on user-selected symptoms.
It uses a supervised learning model — Random Forest Classifier — to find patterns between symptoms and diseases.
""", styles["Body"]))

story.append(Paragraph("2. Project Objectives", styles["Heading"]))
story.append(Paragraph("""
• To develop an intelligent system that predicts diseases using input symptoms.  
• To create a user-friendly web app for interactive prediction.  
• To encourage early awareness and self-education on possible illnesses.  
• To demonstrate AI and ML application in healthcare innovation.
""", styles["Body"]))

# ---------------------------------------------------
# 4️⃣ Project Scope
# ---------------------------------------------------
story.append(Paragraph("3. Project Scope", styles["Heading"]))
story.append(Paragraph("""
This project focuses on disease prediction based solely on symptom data. It does not provide medical diagnosis but rather educational guidance.
It is limited to the diseases included in the dataset and uses textual input through a Streamlit web interface.
""", styles["Body"]))
story.append(PageBreak())

# ---------------------------------------------------
# 5️⃣ Architecture Diagram Page
# ---------------------------------------------------
story.append(Paragraph("4. System Architecture and Data Flow", styles["Heading"]))
story.append(Paragraph("""
The diagram below illustrates the flow of data and processes within the AI Disease Prediction System, from user input to final prediction output.
""", styles["Body"]))
story.append(Spacer(1, 12))

# Create a simple architecture diagram
drawing = Drawing(500, 400)

# Components (rectangles)
components = [
    ("User Interface (Streamlit)", 150, 320, 200, 40, colors.lightblue),
    ("Preprocessing Layer", 150, 250, 200, 40, colors.lightgreen),
    ("ML Model (Random Forest)", 150, 180, 200, 40, colors.orange),
    ("Knowledge Base (CSV Files)", 150, 110, 200, 40, colors.pink),
    ("Output Layer", 150, 40, 200, 40, colors.lavender)
]

for label, x, y, w, h, color in components:
    drawing.add(Rect(x, y, w, h, fillColor=color, strokeColor=colors.black))
    drawing.add(String(x + 10, y + 12, label, fontSize=10))

# Arrows (data flow)
arrows = [
    (250, 320, 250, 290),
    (250, 250, 250, 220),
    (250, 180, 250, 150),
    (250, 110, 250, 80)
]
for x1, y1, x2, y2 in arrows:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.black, strokeWidth=1.5))

story.append(drawing)
story.append(Spacer(1, 24))

story.append(Paragraph("""
**Explanation:**  
1. The user selects symptoms via the Streamlit web interface.  
2. The preprocessing layer encodes the symptoms into a binary vector.  
3. The trained Random Forest model predicts possible diseases.  
4. The knowledge base provides disease descriptions and precautions.  
5. Results are displayed back to the user in real time.
""", styles["Body"]))
story.append(PageBreak())

# ---------------------------------------------------
# 6️⃣ Technical Implementation and Code
# ---------------------------------------------------
story.append(Paragraph("5. Technical Implementation", styles["Heading"]))
story.append(Paragraph("""
The model training is handled in `train_model.py`, where the dataset is cleaned, features are selected, and the model is trained and saved as `model.pkl`.  
The web application logic is defined in `app.py`, using Streamlit to interact with users and show prediction results.
""", styles["Body"]))

story.append(Paragraph("6. Model Accuracy and Results", styles["Heading"]))
story.append(Paragraph("""
The model achieved 100% accuracy on test data using RandomForestClassifier.  
Predictions are presented alongside confidence percentages and precautionary details.
""", styles["Body"]))

story.append(Paragraph("7. Conclusion", styles["Heading"]))
story.append(Paragraph("""
The system effectively demonstrates AI-assisted healthcare education through disease prediction from symptoms.  
It is reliable, easy to use, and provides valuable health awareness insights.  
Future extensions may include integration with hospital databases or real-time symptom tracking.
""", styles["Body"]))

story.append(PageBreak())
story.append(Paragraph("Thank you", styles["TitleCenter"]))
story.append(Paragraph("Prepared by Kazeem Ibrahim Opeyemi", styles["Body"]))
story.append(Paragraph("Supervised by Dr. Jet", styles["Body"]))
story.append(Paragraph("© Abiola Ajimobi Technical University — 2024/2025", styles["Body"]))

# ---------------------------------------------------
# 7️⃣ Build PDF
# ---------------------------------------------------
doc.build(story)
print(f"✅ Speaker Notes (with diagram) generated successfully: {pdf_filename}")
