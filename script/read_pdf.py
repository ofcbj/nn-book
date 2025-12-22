import PyPDF2
import sys

# UTF-8로 출력하도록 설정
sys.stdout.reconfigure(encoding='utf-8')

with open('nn.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    print(f'총 페이지 수: {len(reader.pages)}\n')
    
    # 처음 10페이지 내용 추출
    for i in range(min(10, len(reader.pages))):
        print(f'\n{"="*60}')
        print(f'페이지 {i+1}')
        print(f'{"="*60}\n')
        text = reader.pages[i].extract_text()
        print(text)
