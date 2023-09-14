import streamlit as st
import pandas as pd

############################################################################################
#app():
st.title("Collaborative Filtering")
Review = pd.read_csv("ReviewRaw.csv")
Product = pd.read_csv('ProductRaw.csv', encoding="utf8")
pd.set_option('display.max_colwidth', None) # need this option to make sure cell content is displayed in full
Product['short_name'] = Product['name'].str.split('-').str[0]

############################################################################################

new_user_recs = pd.read_parquet('Tiki_U.parquet')
df_product_product_idx = pd.read_parquet('Tiki_P.parquet')
############################################################################################

# Define functions
#TAKE URL OF AN IMAGE 
def fetch_image(idx):
    selected_product = Product['image'].iloc[[idx]].reset_index(drop=True)
    url = selected_product[0]
    return url

############################################################################################

def get_user_recommendations(customer_id, new_user_recs, df_product_product_idx, min_rating=None):
    find_user_rec = new_user_recs.loc[new_user_recs['customer_id'] == customer_id]
    if not find_user_rec.empty:
        user = find_user_rec.iloc[0]
    # Rest of your code for processing the user recommendations
    else:
        st.warning("Không tìm thấy thông tin đề xuất cho customer id này.")
    user = find_user_rec.iloc[0]
    lst = []
    
    for row in user['recommendations']:   
        row_f = df_product_product_idx.loc[df_product_product_idx.product_id_idx == row['product_id_idx']] 
        row_f_first = row_f.iloc[0]
        lst.append((row['product_id_idx'], row_f_first['product_id'], row['rating']))
        
    dic_user_rec = {'customer_id': user.customer_id, 'recommendations': lst}
    # Lấy danh sách các sản phẩm đề xuất
    recommended_products = dic_user_rec['recommendations']
    # Tạo danh sách sản phẩm đề xuất kèm tên
    recommended_with_names = []
    for rec in recommended_products:
        product_id_idx = rec[1]  # Lấy mã sản phẩm từ recommendation
        product_row = Product[Product['item_id'] == product_id_idx]
        if not product_row.empty:
            product_name = product_row.iloc[0]['short_name']
            product_image = product_row.iloc[0]['image']
            product_rating = round(product_row.iloc[0]['rating'], 2)  # Lấy giá trị rating từ cột 'rating' trong dataframe 'Product' và làm tròn
            product_price = product_row.iloc[0]['price']  # Lấy giá sản phẩm từ cột 'price' trong dataframe 'Product'

            # Kiểm tra rating của sản phẩm
            if min_rating is not None and product_rating < min_rating:
                continue

            recommended_with_names.append((rec[0], product_name, product_rating, product_price, product_image))
    
    return recommended_with_names

############################################################################################

# Input customer id
number = st.number_input("Nhập customer id (ví dụ: 494671, 709310, 10701688, 9909549...):", min_value=0)
st.write("Customer id bạn nhập là: ", number)


# Check if customer id exists
if number not in new_user_recs['customer_id'].unique():
    st.warning("Có thể bạn thích những sản phẩm này")
    
    # Get top 10 products with the most reviews
    top_products = Review['product_id'].value_counts().nlargest(10).index.tolist()
    
    # Fetch product information
    top_product_info = []
    for product_id in top_products:
        product_row = Product[Product['item_id'] == product_id]
        if not product_row.empty:
            product_rating = round(product_row.iloc[0]['rating'], 1)  # Lấy giá trị rating và làm tròn
            if product_rating >= 4.5:
                product_name = product_row.iloc[0]['short_name']
                product_image = product_row.iloc[0]['image']
                product_price = '{:,.0f}đ'.format(product_row.iloc[0]['price'])  # Định dạng giá sản phẩm
                top_product_info.append((product_name, product_image, product_rating, product_price))
    
    # Display top products with price and rating
    cols = st.columns(len(top_product_info))
    for c in range(len(top_product_info)):
        with cols[c]:
            st.image(top_product_info[c][1], caption=top_product_info[c][0])
            st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Giá:</span> <span style='color: #CC0000;'>{top_product_info[c][3]}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Đánh giá:</span> <span style='color: #CC0000;'>{top_product_info[c][2]}⭐</span></div>", unsafe_allow_html=True)
else:
    # Choose maximum number of products that system will recommend
    n = st.slider(
        'Chọn số lượng sản phẩm tối đa mà bạn muốn hệ thống giới thiệu (từ 1 đến 10)',
        1, 10, 5)
        # Create list icon star
    star_icons = ["⭐", "⭐⭐ trở lên", "⭐⭐⭐ trở lên", "⭐⭐⭐⭐ trở lên", "⭐⭐⭐⭐⭐"]

    # Select the product rating you want to recommend
    min_rating_index = st.select_slider(
        'Đánh giá', 
        options=list(range(5)), 
        format_func=lambda x: star_icons[x],
        value=0)

    min_rating = min_rating_index + 1
    if min_rating < 5:
        st.write('Các sản phẩm có đánh giá', min_rating,'⭐','trở lên')
    else:
        st.write('Các sản phẩm có đánh giá', min_rating,'⭐')


    # 'Recommend' button
    if st.button('Recommend'):
        recommendations = get_user_recommendations(customer_id=number, new_user_recs=new_user_recs, df_product_product_idx=df_product_product_idx, min_rating=min_rating)
        recommendations = recommendations[:n]

        num_items_per_row = 5  # Số sản phẩm trên mỗi dòng
        num_rows = (n + num_items_per_row - 1) // num_items_per_row

        for row in range(num_rows):
            cols = st.columns(num_items_per_row)

            for col in range(num_items_per_row):
                index = row * num_items_per_row + col
                if index < n and index < len(recommendations):
                    with cols[col]:
                        st.image(recommendations[index][4], caption=recommendations[index][1])
                        formatted_price = '{:,.0f}đ'.format(recommendations[index][3])  # Định dạng giá sản phẩm
                        st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Giá:</span> <span style='color: #CC0000;'>{formatted_price}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Đánh giá:</span> <span style='color: #CC0000;'>{recommendations[index][2]}⭐</span></div>", unsafe_allow_html=True)

        st.write('Các sản phẩm customer id:', number, 'đã mua gần đây:')
        products_bought_by_customer = Review.loc[(Review['customer_id'] == number) & (Review['rating'] >= 4)]
        if not products_bought_by_customer.empty:
            product1 = Product[['item_id', 'price', 'brand', 'group', 'image', 'short_name']]
            # Lấy danh sách 10 sản phẩm đã mua gần đây có rating >= 4
            recent_products = products_bought_by_customer[products_bought_by_customer.duplicated('product_id', keep='first') == False].head(10)
            
            # Thực hiện phép join giữa Product và recent_products để lấy thông tin tên sản phẩm, giá và rating
            result_df = recent_products.merge(product1, left_on='product_id', right_on='item_id', how='inner')
            result_df['name_rating'] = result_df.apply(lambda row: f"{row['short_name']}", axis=1)
            result_df = result_df[["product_id", "short_name", "rating", 'image', 'name_rating', 'price']]
            names = result_df['name_rating'].values.tolist()
            images = result_df['image'].values.tolist()
            prices = result_df['price'].values.tolist()
            
            # Chia danh sách sản phẩm đã mua thành 2 dòng (mỗi dòng 5 sản phẩm)
            for j in range(0, len(names), 5):
                current_names = names[j:j+5]
                current_images = images[j:j+5]
                current_prices = prices[j:j+5]
                current_names = current_names[:5]
                current_images = current_images[:5]
                current_prices = current_prices[:5]
                cols = st.columns(len(current_names))
                for c in range(len(current_names)):
                    with cols[c]:
                        st.image(current_images[c], caption=current_names[c])
                        formatted_price = '{:,.0f}đ'.format(current_prices[c])  # Định dạng giá sản phẩm
                        st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Giá:</span> <span style='color: #CC0000;'>{formatted_price}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 14px; color: #333; text-align: center;'><span style='color: white;'>Đánh giá:</span> <span style='color: #CC0000;'>{result_df['rating'].iloc[c]:.1f}⭐</span></div>", unsafe_allow_html=True)
