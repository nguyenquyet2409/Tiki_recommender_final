import streamlit as st
from PIL import Image


st.markdown("# Project Overview")
st.header("I. Project Objectives")
"""
Tiki là một hệ sinh thái thương mại “all in one”, trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á.

Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?

Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng. 
=> Xây dựng các mô hình đề xuất:▪ Content-based filtering ▪ Collaborative filtering
"""
"\n"
st.header("II. Overview")
"""
Recommender system là các thuật toán nhằm đề xuất các item có liên quan cho người dùng 
(Item có thể là phim để xem, văn bản để đọc, sản phẩm cần mua hoặc bất kỳ thứ gì khác tùy thuộc 
vào ngành dịch vụ).
Recommender system thực sự quan trọng trong một số lĩnh vực vì chúng có thể tạo ra một khoản thu nhập khổng lồ hoặc 
cũng là một cách để nổi bật đáng kể so với các đối thủ cạnh tranh.

Có hai recommender system phổ biến nhất là Collaborative Filtering (CF) và Content-Based
"""
"\n"
image = Image.open('recommender system.png')

st.image(image, caption='Two main types of Recommender System')
"\n"
st.subheader("Collaborative Filtering Methods")
"""
Phương pháp này dựa hoàn toàn vào các tương tác trước đây được ghi lại giữa người dùng và các mục để tạo ra các gợi ý mới. 
Các tương tác này được lưu trữ trong ma trận "user-item interactions matrix". 
Các phương pháp cộng tác sử dụng để phát hiện ra các người dùng tương tự và/hoặc các mục tương tự 
và đưa ra các dự đoán dựa trên sự tương tự được ước tính này.
"""
"\n"
st.subheader("Content Based Methods")
"""
Không giống như các phương pháp cộng tác chỉ dựa vào tương tác người dùng-mục, các phương pháp dựa trên nội dung sử dụng thông tin bổ sung về người dùng 
và các mục. Ý tưởng của các phương pháp dựa trên nội dung là cố gắng xây dựng một mô hình, dựa trên các "đặc điểm" có sẵn, 
để giải thích các tương tác người dùng-mục đã quan sát được.

"""
"\n"

st.header("III. Demo App")
"""
Ứng dụng minh họa này được xây dựng cho cả 2 phương pháp là:  Content Based Method và Collaborative Filtering Method.

1. Thuật toán được lựa chọn là Gensim vì hiệu suất của có thời gian xử lý nhanh, lưu trữ tốn ít bộ nhớ. 
Ứng dụng cũng cung cấp hai cách tiếp cận cho người dùng khi sử dụng Gensim:
- Bằng cách chọn sản phẩm từ danh sách cố định và yêu cầu hệ thống đề xuất.
- Bằng cách nhập mô tả sản phẩm và yêu cầu hệ thống đề xuất.

2. Collaborative Filtering Method
- Bằng cách chọn customer id để yêu cầu hệ thống đề xuất.
"""
"\n"
