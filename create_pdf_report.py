from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils

try:
    # Create a PDF file
    pdf_file = "confussion_matrix.pdf"

    # Create a canvas
    c = canvas.Canvas(pdf_file, pagesize=letter)

    # Set the position for the image
    x, y = 100, 500

    # Add text to the canvas
    text = "Hello, World!"
    c.setFont("Helvetica", 12)
    c.drawString(x, y, text)

    # Set the position and size for the image
    image_x, image_y = 100, 300
    image_width, image_height = 200, 100

    # Load the image file (replace 'image.png' with your image file path)
    image_path = 'practice_models/confusionmatrix.png'
    img = utils.ImageReader(image_path)

    # Add the image to the canvas
    c.drawImage(img, image_x, image_y, width=image_width, height=image_height)

    # Save the PDF
    c.save()

    print(f"PDF with image created successfully: {pdf_file}")
except Exception as e:
    print(f"An error occurred: {e}")
