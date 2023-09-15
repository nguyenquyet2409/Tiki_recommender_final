import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
from gensim import corpora, models, similarities
import re
import seaborn as sns
import streamlit as st
import pickle


################################################################################################
page_bg_img = '''
<style>

[data-testid="stSidebar"] {
background: linear-gradient(-225deg, #5271C4 0%, #B19FFF 48%, #ECA1FE 100%);
}

</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

############################################################################################
st.title("Content Based Filtering")
Product = pd.read_csv('Product_clean.csv', encoding="utf8", index_col=0)
pd.set_option('display.max_colwidth', None) # need this option to make sure cell content is displayed in full
Product['short_name'] = Product['name'].str.split('-').str[0]
product_map = Product.iloc[:,[0,-1]]
product_list = product_map['short_name'].values

############################################################################################

# Define functions to use for both methods
##### TEXT PROCESSING #####
def process_text(document):
    # Change to lower text
    document = document.lower()
    # Remove HTTP links
    document = document.replace(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '')
    # Remove line break
    document = document.replace(r'[\r\n]+', ' ')
    # Change / by white space
    document = document.replace('/', ' ') 
    # Change , by white space
    document = document.replace(',', ' ') 
    # Remove punctuations
    document = document.replace('[^\w\s]', '')
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        document = document.replace(char, '')
    # Replace mutiple spaces by single space
    document = document.replace('[\s]{2,}', ' ')
    # Word_tokenize
    document = word_tokenize(document, format="text")   
    # Pos_tag
    document = pos_tag(document)    
    # Remove stopwords
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'   
    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    stop_words = stop_words.split()  
    document = [[word[0] for word in document if not word[0] in stop_words]] 
    return document

##### TAKE URL OF AN IMAGE #####
def fetch_image(idx):
    selected_product = Product['image'].iloc[[idx]].reset_index(drop=True)
    url = selected_product[0]
    return url

##### CHECK PRODUCT SIMILARITIES BY GENSIM MODEL AND RETURN NAMES & IMAGES OF TOP PRODUCTS WITH HIGHEST SIMILARITY INDEX #####

with open('dictionary.pkl', 'rb') as file:
    dictionary = pickle.load(file)
tfidf = models.tfidfmodel.TfidfModel.load("tfidf.tfidfmodel")
index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")


##############################################################

def show_similar_products(dataframe, index, selected_product_index, num_similar, min_rating=None):
    # Chọn sản phẩm đang xem dựa trên chỉ số
    product_selection = dataframe.iloc[[selected_product_index]]
    
    # Lấy thông tin sản phẩm
    name_description_pre = product_selection['content'].to_string(index=False)
    view_product = name_description_pre.lower().split()
    
    # Chuyển từ khóa tìm kiếm thành Sparse Vectors
    kw_vector = dictionary.doc2bow(view_product)
    
    # Tính toán độ tương tự
    sim = index[tfidf[kw_vector]]
    
    # Sắp xếp danh sách sim theo thứ tự giảm dần
    sorted_sim_indices = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)
    
    # Tạo DataFrame chứa thông tin các sản phẩm tương tự
    similar_products_info = []
    for i in range(num_similar+1):
        index = sorted_sim_indices[i]
        similarity = sim[index]
        similar_product = dataframe.iloc[[index]]
        
        # Check rating filter
        product_rating = similar_product['rating'].values[0]
        
        # Kiểm tra xem sản phẩm có đánh giá lớn hơn hoặc bằng min_rating không
        if min_rating is not None and product_rating < min_rating:
            continue
        
        similar_product_name = similar_product['short_name'].values[0]  # Lấy tên sản phẩm
        similar_product_price = similar_product['price'].values[0]  # Lấy giá sản phẩm
        similar_product_rating = similar_product['rating'].values[0]  # Lấy đánh giá sản phẩm
        similar_product_image = similar_product['image'].values[0]  # Lấy đường dẫn hình ảnh sản phẩm
        similar_product_info = {
            "index": similar_product.index.values[0],
            "name": f"{similar_product_name}",
            "price": f"{similar_product_price}",
            "rating": f"{similar_product_rating}",
            "score": f"{similarity:.2f}",
            "image": f"{similar_product_image}" 
        }
        similar_products_info.append(similar_product_info)
    
    result = pd.DataFrame(similar_products_info)
    
    # Check if there are any products to recommend
    if len(result) == 0:
        st.warning("Không có sản phẩm phù hợp với tiêu chí đã chọn.")
        return [], [], [], []
    
    n_highest_score = result.sort_values(by='score', ascending=False).head(num_similar)
    
    # Extract product_id of above request
    id_tolist = list(n_highest_score['index'])
    recommended_names = []
    recommended_prices = []
    recommended_ratings = []
    recommended_images = []
    for i in id_tolist:
        # Fetch the product names
        product_name = dataframe['name'].iloc[[i]]
        recommended_names.append(product_name.to_string(index=False))
        # Fetch the product prices
        product_price = dataframe['price'].iloc[[i]]
        recommended_prices.append(product_price.to_string(index=False))
        # Fetch the product ratings
        product_rating = dataframe['rating'].iloc[[i]]
        recommended_ratings.append(product_rating.to_string(index=False))
        # Fetch the product images
        recommended_images.append(fetch_image(i))
    
    return recommended_names, recommended_prices, recommended_ratings, recommended_images

############################################################################################

# Define separate page to demo each method

def filter_list():
    # Markdown name of Content_based method
    st.markdown("### By Product List")

    # Select product from list
    selected_idx = st.selectbox("Chọn sản phẩm muốn xem: ", range(len(product_list)), format_func=lambda x: product_list[x])

    # Fetch image of selected product
    idx = selected_idx
    st.image(fetch_image(idx))

    # Choose maximum number of products that system will recommend
    n = st.slider(
    'Chọn số lượng sản phẩm tối đa tương tự như trên mà bạn muốn hệ thống giới thiệu (từ 1 đến 10)',
    1, 10, 5)
#########################################################################################################################
    # # Create list icon star
    # star_icons = ["⭐", "⭐⭐ trở lên", "⭐⭐⭐ trở lên", "⭐⭐⭐⭐ trở lên", "⭐⭐⭐⭐⭐"]

    # # Select the product rating you want to recommend
    # min_rating_index = st.select_slider(
    #     'Đánh giá', 
    #     options=list(range(5)), 
    #     format_func=lambda x: star_icons[x],
    #     value=3)

    # min_rating = min_rating_index + 1
    # if min_rating < 5:
    #     st.write('Các sản phẩm có đánh giá', min_rating,'⭐','trở lên')
    # else:
    #     st.write('Các sản phẩm có đánh giá', min_rating,'⭐')

###########################################################################################################################        
    # Create list icon star
    star_icons = ["⭐ trở lên", "⭐⭐ trở lên", "⭐⭐⭐ trở lên", "⭐⭐⭐⭐ trở lên", "⭐⭐⭐⭐⭐"]

    # Radio button for min_rating
    selected_min_rating = st.radio("Chọn đánh giá:", star_icons)

    # Sử dụng index của dòng được chọn để xác định min_rating
    if selected_min_rating:
        min_rating = star_icons.index(selected_min_rating) + 1

    if min_rating <5:
        st.write(f'Các sản phẩm có đánh giá tối thiểu {min_rating}⭐ trở lên')
    else:
        st.write(f'Các sản phẩm có đánh giá tối thiểu {min_rating}⭐')
# 'Recommend' button
    if st.button('Recommend'):
        min_rating = min_rating
        index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")
        names, prices, ratings, images = show_similar_products(dataframe=Product, index=index, selected_product_index=idx, num_similar=n+3, min_rating=min_rating)

        if len(names) > 0 and len(images) > 0:
            # Lọc ra các sản phẩm có rating >= min_rating
            filtered_names = []
            filtered_prices = []
            filtered_ratings = []
            filtered_images = []

            for i in range(1, len(names) - 1):  # Bỏ qua sản phẩm đầu và cuối
                product_rating = float(ratings[i])
                if product_rating >= min_rating:
                    filtered_names.append(names[i])
                    filtered_prices.append(prices[i])
                    filtered_ratings.append(ratings[i])
                    filtered_images.append(images[i])

            num_display = min(n, len(filtered_names))  # Số sản phẩm cần hiển thị
            items_per_row = 5  # Số sản phẩm trong mỗi hàng
            num_rows = num_display // items_per_row + (num_display % items_per_row > 0)

            # Sử dụng st.columns để tạo cột cho từng sản phẩm
            for i in range(num_rows):
                cols = st.columns(items_per_row)
                start_idx = i * items_per_row
                end_idx = (i + 1) * items_per_row if i < num_rows - 1 else num_display

                for c in range(start_idx, end_idx):
                    with cols[c % items_per_row]:
                        st.image(filtered_images[c], caption=filtered_names[c])
                        st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Giá:</span> <span style='color: #CC0000;'>{filtered_prices[c]}đ</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Đánh giá:</span> <span style='color: #CC0000;'>{filtered_ratings[c]}⭐</span></div>", unsafe_allow_html=True)

#### CONTENT_BASED FILTERING BY INPUTING DESCRIPTION #####
def input_description_product(dataframe, index, input_product_name, num_similar, min_rating=None):
    # Chuyển từ khóa tìm kiếm thành Sparse Vectors
    view_product = input_product_name.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    
    # Tính toán độ tương tự
    sim = index[tfidf[kw_vector]]
    
    # Sắp xếp danh sách sim theo thứ tự giảm dần
    sorted_sim_indices = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)
    
    # Tạo DataFrame chứa thông tin các sản phẩm tương tự
    similar_products_info = []
    for i in range(num_similar+1):
        index = sorted_sim_indices[i]
        similarity = sim[index]
        similar_product = dataframe.iloc[[index]]
        similar_product_name = similar_product['short_name'].values[0]  # Lấy tên sản phẩm
        similar_product_price = similar_product['price'].values[0]  # Lấy giá sản phẩm
        similar_product_rating = similar_product['rating'].values[0]  # Lấy đánh giá sản phẩm
        similar_product_image = similar_product['image'].values[0]  # Lấy đường dẫn hình ảnh sản phẩm
        
        # Kiểm tra xem sản phẩm có đánh giá lớn hơn hoặc bằng min_rating không
        if min_rating is not None and similar_product_rating < min_rating:
            continue
        
        similar_product_info = {
            "index": similar_product.index.values[0],
            "name": f"{similar_product_name}",
            "price": f"{similar_product_price}",
            "rating": f"{similar_product_rating}",
            "score": f"{similarity:.2f}",
            "image": f"{similar_product_image}" 
        }
        similar_products_info.append(similar_product_info)
    
    result = pd.DataFrame(similar_products_info)
    if len(result) == 0:
        st.warning("Không có sản phẩm phù hợp với tiêu chí đã chọn.")
        return [], [], [], []
    n_highest_score = result.sort_values(by='score', ascending=False).head(num_similar)
    # Extract product_id of above request
    id_tolist = list(n_highest_score['index'])
    recommended_names = []
    recommended_prices = []
    recommended_ratings = []
    recommended_images = []
    for i in id_tolist:
        # Fetch the product names
        product_name = dataframe['name'].iloc[[i]]
        recommended_names.append(product_name.to_string(index=False))
        # Fetch the product prices
        product_price = dataframe['price'].iloc[[i]]
        recommended_prices.append(product_price.to_string(index=False))
        # Fetch the product ratings
        product_rating = dataframe['rating'].iloc[[i]]
        recommended_ratings.append(product_rating.to_string(index=False))
        # Fetch the product images
        recommended_images.append(fetch_image(i))
    return recommended_names, recommended_prices, recommended_ratings, recommended_images


def input_description():
    # Markdown name of Content_based method
    st.markdown("### By Inputing Description")

    # input product description
    text_input = st.text_input(
        "Nhập mô tả sản phẩm để tìm kiếm: "
    )
    if text_input:
        st.write("Mô tả sản phẩm của bạn: ", text_input)
    # Choose maximum number of products that system will recommend
    n = st.slider(
    'Chọn số lượng sản phẩm tối đa tương tự như trên mà bạn muốn hệ thống giới thiệu (từ 1 đến 10)',
    1, 10, 5)

################################################################### Slicer#################################
   
    # # Create list icon star
    # star_icons = ["⭐", "⭐⭐ trở lên", "⭐⭐⭐ trở lên", "⭐⭐⭐⭐ trở lên", "⭐⭐⭐⭐⭐"]

    # # Select the product rating you want to recommend
    # min_rating_index = st.select_slider(
    #     'Đánh giá', 
    #     options=list(range(5)), 
    #     format_func=lambda x: star_icons[x],
    #     value=3)

    # min_rating = min_rating_index + 1

    # if min_rating < 5:
    #     st.write('Các sản phẩm có đánh giá', min_rating,'⭐','trở lên')
    # else:
    #     st.write('Các sản phẩm có đánh giá', min_rating,'⭐')
    

################################################### Radio button #################################
    # Create list icon star
    star_icons = ["⭐ trở lên", "⭐⭐ trở lên", "⭐⭐⭐ trở lên", "⭐⭐⭐⭐ trở lên", "⭐⭐⭐⭐⭐"]

    # Radio button for min_rating
    selected_min_rating = st.radio('Chọn đánh giá',star_icons)

    # Sử dụng index của dòng được chọn để xác định min_rating
    if selected_min_rating:
        min_rating = star_icons.index(selected_min_rating) + 1

    if min_rating <5:
        st.write(f'Các sản phẩm có đánh giá tối thiểu {min_rating}⭐ trở lên')
    else:
        st.write(f'Các sản phẩm có đánh giá tối thiểu {min_rating}⭐')


    # 'Recommend' button
    if st.button('Recommend'):
        index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")
        names, prices, ratings, images = input_description_product(dataframe=Product, index=index, input_product_name=text_input, num_similar=n+1, min_rating=min_rating)
        names = names[:n]
        prices = [f"{float(price):,.0f}" for price in prices[:n]]
        ratings = ratings[:n]
        images = images[:n]

        num_items_per_row = 5  # Số sản phẩm trên mỗi dòng
        num_rows = (n + num_items_per_row - 1) // num_items_per_row

        for row in range(num_rows):
            cols = st.columns(num_items_per_row)

            for col in range(num_items_per_row):
                index = row * num_items_per_row + col
                if index < n:
                    with cols[col]:
                        if index < len(images) and index < len(names):
                            st.image(images[index], caption=names[index])
                            st.markdown(f"<div style='font-size: 14px; color: white; text-align: center;'>Giá: <span style='color: #CC0000;'>{prices[index]}đ</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size: 14px; color: white; text-align: center;'>Đánh giá: <span style='color: #CC0000;'>{ratings[index]}⭐</span></div>", unsafe_allow_html=True)                   

##### CALLING PAGE  #####
page_names_to_funcs = {
    "Chọn sản phẩm": filter_list,
    "Tìm sản phẩm bằng cách nhập mô tả": input_description
    }
selected_page = st.sidebar.selectbox("Chọn hình thức gợi ý sản phẩm", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
