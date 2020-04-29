# import PyPDF2
#
# # creating a pdf file object
# pdfFileObj = open('CV.pdf', 'rb')
#
# # creating a pdf reader object
# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
#
# # printing number of pages in pdf file
# print(pdfReader.numPages)
#
# # creating a page object
# pageObj = pdfReader.getPage(0)
#
# # extracting text from page
# print(pageObj.extractText())
#
# # closing the pdf file object
# pdfFileObj.close()
import PyPDF2

def getPDFContent(path):
    content = ""
    # Load PDF into pyPDF
    pdf = PyPDF2.PdfFileReader(open(path, "rb"))
    # Iterate pages
    for i in range(0, pdf.getNumPages()):
        # Extract text from page and add to content
        content += pdf.getPage(i).extractText() + "\n"
    # Collapse whitespace
    content = " ".join(content.replace("\xa0", " ").strip().split())
    return content

print (getPDFContent("docs/CV.pdf"))