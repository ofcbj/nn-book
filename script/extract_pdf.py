import PyPDF2

with open('nn.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    total_pages = len(reader.pages)
    
    with open('nn_extracted.txt', 'w', encoding='utf-8') as out:
        out.write(f'총 페이지 수: {total_pages}\n\n')
        
        # 모든 페이지 내용 추출
        for i in range(total_pages):
            out.write(f'\n{"="*60}\n')
            out.write(f'페이지 {i+1}\n')
            out.write(f'{"="*60}\n\n')
            text = reader.pages[i].extract_text()
            out.write(text)
            out.write('\n')

print(f'추출 완료! 총 {total_pages} 페이지')
