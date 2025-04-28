# 대화형 챗봇 애플리케이션에 필요한 라이브러리 임포트
import streamlit as st  # type: ignore # 웹 페이지를 쉽게 만들 수 있는 라이브러리
import json             # JSON 데이터 처리 라이브러리
import chromadb         # 확장성 있는 벡터 데이터베이스 라이브러리
from chromadb.utils import embedding_functions # 텍스트를 숫자 벡터로 변환하는 함수
from openai import OpenAI  # OpenAI API 사용 라이브러리
import re             # 정규표현식 라이브러리
from PIL import Image    # 이미지 처리 라이브러리
import folium          # 지도 시각화 라이브러리
import pandas as pd      # 데이터 분석 라이브러리
from streamlit_folium import st_folium # Streamlit에서 Folium 지도를 사용하기 위한 라이브러리
from folium import IFrame  # Folium에서 HTML 내용을 표시하기 위한 라이브러리

# --------------------------------->
# 페이지 설정
# --------------------------------->
st.set_page_config(
    page_title="광진구 착한가격 추천 챗봇",
    page_icon="🕳️",
    layout="wide"
)

# --------------------------------->
# 데이터 로드 및 전처리
# --------------------------------->
df = pd.read_csv(r"C:\Users\itwill\Downloads\광진구_추천업소_전처리_위경도추가.csv")
# 위도, 경도 결측치 제거
df = df.dropna(subset=['latitude', 'longitude'])

# --------------------------------->
# ChromaDB 관련 함수 정의
# --------------------------------->
# ChromaDB 클라이언트 초기화 (캐싱 적용)
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="c:\\auto_excel\\chroma_db")

# ChromaDB 클라이언트 가져오기 (캐싱 적용)
@st.cache_resource
def get_chroma_client():
    return init_chroma_client()

# 사용 가능한 컬렉션 이름 목록 가져오기
def get_available_collections():
    try:
        client = get_chroma_client()
        # 컬렉션 객체에서 이름만 추출하여 리스트로 반환
        collection_names = [collection.name for collection in client.list_collections()]
        return collection_names
    except Exception as e:
        st.sidebar.error(f"컬렉션 목록 로드 오류: {e}")
        return []

# --------------------------------->
# 메인 화면 UI
# --------------------------------->
st.title("착한가격 업소 데이터 분석가")
st.write("착한가격 업소 데이터에 대해 질문해보세요.")


# --------------------------------->
# 사이드바 설정 UI
# --------------------------------->
with st.sidebar:
    # 이미지 표시
    try:
        image = Image.open(r"C:\Users\itwill\Downloads\sub0402_img01.png")
        st.image(image)
    except FileNotFoundError:
        st.warning("사이드바 이미지를 찾을 수 없습니다.")
   

    # 설정 섹션
    collection_name = "gjg_report"
    # 사용 가능한 컬렉션을 불러오되, 'gjg_report'만 남기기
    all_cols = get_available_collections()
    allowed = [c for c in all_cols if c == "gjg_report"]
    
    # 설정 섹션
    st.title("설정")
    api_key = st.text_input("OpenAI API 키를 입력하세요", type="password")

    # # 컬렉션 선택 드롭다운
    # collection_list = get_available_collections()
    # collection_name = None   

    # if allowed:
    #     collection_name = st.selectbox(
    #         "사용할 컬렉션 선택",
    #         allowed,
    #         index=0
    #     )
    # else:
    #     st.warning("사용 가능한 컬렉션이 없습니다.")   

    # 지도 필터링 UI
    st.sidebar.markdown("---")
    st.sidebar.subheader("지도 필터")
    selected_category = st.sidebar.multiselect("업종 선택", df['업종'].unique())
    상품권_선택 = st.sidebar.selectbox("서울사랑상품권 가맹", ['전체', '가능', '불가능'])

# --------------------------------->
# 지도 필터링 로직
# --------------------------------->
filtered_df = df.copy()
filters_applied = False  # 필터가 적용되었는지 여부를 추적하는 변수

if selected_category:
    filtered_df = filtered_df[df['업종'].isin(selected_category)]
    filters_applied = True

if 상품권_선택 != '전체':
    filtered_df = filtered_df[filtered_df['사랑상품권'] == 상품권_선택]
    filters_applied = True

# --------------------------------->
# 지도 시각화 UI
# --------------------------------->
st.markdown("### 추천업소 지도")
center = [37.5502596, 127.073139]
m = folium.Map(location=center, zoom_start=16)

group_착한 = folium.FeatureGroup(name="착한업소").add_to(m)
group_비착한 = folium.FeatureGroup(name="일반업소").add_to(m)

# 필터가 선택되지 않은 경우 광진구 마커만 표시
if not filters_applied:
    folium.Marker(
        location=center,
        popup="광진구 중심",
        tooltip="광진구",
        icon=folium.Icon(color="red")
    ).add_to(m)
elif filtered_df.empty:
    st.warning("조건에 맞는 업소가 없습니다. 필터를 조정해주세요.")
else:
    for idx, row in filtered_df.iterrows():
        html = f"""
        <b>업체명:</b> {row['사업장명']}<br>
        <b>업종:</b> {row['업종']}<br>
        <b>착한업소:</b> {row['착한업소']}<br>
        <b>상품권 가맹:</b> {row['사랑상품권']}<br>
        <b>위생등급:</b> {row['위생등급'] if pd.notna(row['위생등급']) else '미지정'}<br>
        <b>행정처분:</b> {row['행정처분'] if pd.notna(row['행정처분']) else '없음'}
        """
        iframe = IFrame(html, width=300, height=200)
        popup = folium.Popup(iframe, max_width=300)
        tooltip = row['사업장명']
        marker = folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            tooltip=tooltip,
            icon=folium.Icon(color="blue")
        )
        marker.add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=900, height=600)

# 벡터 데이터베이스에서 컬렉션 가져오기
def get_collection(collection_name):
    if not collection_name:
        return None
    try:
        client = init_chroma_client()
        collection = client.get_collection(name=collection_name)
        return collection
    except Exception as e:
        st.error(f"컬렉션 가져오기 오류: {e}")
        return None

# --------------------------------->
# 벡터 데이터베이스 검색 함수
# --------------------------------->
def search_vector_db(collection, query, n_results=20):
    try:
        if not collection:
            return [{"content": "컬렉션을 불러올 수 없습니다. 컬렉션을 선택해주세요.", "title": "오류", "metadata": {}}]

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        documents = []
        for i in range(len(results['documents'][0])):
            document = {
                "content": results['documents'][0][i],
                "title": results['metadatas'][0][i].get('title', '제목 없음'),
                "metadata": results['metadatas'][0][i]
            }
            documents.append(document)

        return documents
    except Exception as e:
        st.sidebar.error(f"검색 오류: {e}")
        return [{"content": f"검색 중 오류 발생: {e}", "title": "오류", "metadata": {}}]

# --------------------------------->
# OpenAI를 활용한 응답 생성 함수
# --------------------------------->
def get_gpt_response(query, search_results, api_key, model="gpt-4o-mini"):
    if not api_key:
        return "OpenAI API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해주세요."

    try:
        api_key = api_key.replace('\ufeff', '')
        client = OpenAI(api_key=api_key)

        context = "다음은 광진구 착한가격 업소 관련 데이터입니다:\n\n"
        for i, result in enumerate(search_results):
            context += f"업소 {i+1}:\n"
            context += f"업소명: {result['title']}\n"
            if result['metadata']:
                metadata = result['metadata']
                # 메타데이터에 작성일이 있으면 추가
                if 'published_date' in metadata:
                    context += f"등록일: {metadata['published_date']}\n"
                # URL이나 출처가 있으면 추가
                if 'url' in metadata:
                    context += f"출처: {metadata['url']}\n"
                if 'source' in metadata:
                    context += f"제공처: {metadata['source']}\n"
            # 본문 내용
            content = result['content']
            if len(content) > 8000:
                content = content[:8000] + "..."
            context += f"설명: {content}\n\n"

        system_prompt = """당신은 착한가격업소(가성비 좋은 업소) 발굴 및 분석 전문가입니다. 제공된 문서들을 바탕으로 사용자 질문에
        정확하고 간결하게 답변해주세요.

        답변 작성 가이드라인:
        1. 사용자의 질문에 직접적으로 관련된 내용만 답변하세요.
        2. 질문과 관련 없는 부가정보나 제안사항은 포함하지 마세요.
        3. 제공된 문서에서 확인할 수 있는 사실에 기반하여 답변하세요.
        4. 답변은 간결하고 명확하게 작성하세요.
        5. 사용자가 광진구나 특정 지역에 대해 질문하면, 해당 지역의 착한가격업소 정보에 집중하여 답변하세요.
        """

        user_prompt = f"""{context}

        사용자 질문: {query}

        위 문서들을 분석하여 사용자의 질문에만 직접적으로 답변해주세요.
        질문에 관련된 정보만 제공하고, 불필요한 배경설명이나 추가 정보는 생략해주세요.
        """

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "auth" in error_msg.lower() or "api key" in error_msg.lower():
            return "OpenAI API 키 인증에 실패했습니다. API 키를 확인해주세요."
        else:
            return f"분석 중 오류가 발생했습니다: {error_msg}"

# --------------------------------->
# 간단한 응답 생성 함수 (API 키 없을 때)
# --------------------------------->
def get_simple_response(query, search_results):
    if not search_results or search_results[0].get("title") == "오류":
        return "관련 데이터를 찾을 수 없습니다."

    result_text = f"'{query}'에 대한 검색 결과:\n\n"
    for i, result in enumerate(search_results[:5]):
        result_text += f"**문서 {i+1}:** {result['title']}\n"
        if 'published_date' in result['metadata']:
            result_text += f"**날짜:** {result['metadata']['published_date']}\n"
        content = result['content']
        if len(content) > 150:
            content = content[:150] + "..."
        result_text += f"{content}\n\n"

    result_text += "더 자세한 분석을 위해서는 OpenAI API 키를 입력해주세요."
    return result_text

# --------------------------------->
# 챗봇 응답 생성 함수 (메인 로직)
# --------------------------------->
def chat_response(question, collection):
    # 벡터 데이터베이스 검색
    search_results = search_vector_db(collection, question)

    # ChatGPT API 키가 있으면 GPT 사용, 없으면 간단한 응답
    if api_key:
        return get_gpt_response(question, search_results, api_key)
    else:
        return get_simple_response(question, search_results)

# 컬렉션 가져오기
collection = get_collection(collection_name) # 백터 데이터베이스에서 컬렉션을 가져옴

 # 컬렉션 정보 표시
if collection:
    try:
        count = collection.count()
        st.sidebar.success(f"컬렉션 '{collection_name}'에서 {count}개의 문서를 불러왔습니다.")
    except Exception as e:
        st.sidebar.sidebar.warning(f"컬렉션 정보 확인 중 오류: {e}")
else:
    st.warning(f"컬렉션을 선택하거나 찾을 수 없습니다. 컬렉션 목록을 확인하세요.")

# --------------------------------->
# 세션 상태 관리 및 대화 UI
# --------------------------------->
# 세션 상태 초기화 (대화 기록)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요 (예: 광진구에서 착한 가격으로 식사할 수 있는 곳은 어디인가요?)"):
    # 컬렉션이 없는 경우 오류 메시지 표시
    if not collection:
        with st.chat_message("assistant"):
            st.markdown("⚠️ 컬렉션을 선택해주세요. 현재 컬렉션이 선택되지 않았거나 찾을 수 없습니다.")
        st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ 컬렉션을 선택해주세요. 현재 컬렉션이 선택되지 않았거나 찾을 수 없습니다."})
    else:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # 응답 생성 및 표시
        with st.spinner("질문과 관련된 문서를 수집하여 답변을 준비하고 있는 중..."):
            response = chat_response(prompt, collection)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# --------------------------------->
# 사이드바 예시 질문 및 대화 기록 초기화
# --------------------------------->
st.sidebar.header("예시 질문")
example_questions = [
    "광진구에서 착한가격업소는 어떤 기준으로 선정되나요?",
    "광진구에 새롭게 등록된 착한가격업소가 있나요?",
    "광진구 착한가격업소의 분포는 어떤가요?",
    "착한가격업소와 위생등급 업소의 관계는 어떤가요?",
    "착한가격업소로 등록되면 어떤 혜택이 있나요?"
]

# 예시 질문 버튼
for question in example_questions:
    if st.sidebar.button(question):
        if not collection:
            with st.chat_message("assistant"):
                st.markdown("⚠️ 컬렉션을 선택해주세요. 현재 컬렉션이 선택되지 않았거나 찾을 수 없습니다.")
            st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ 컬렉션을 선택해주세요. 현재 컬렉션이 선택되지 않았거나 찾을 수 없습니다."})
        else:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("질문과 관련된 문서를 수집하여 분석하고 있는 중..."):
                response = chat_response(question, collection)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

# 대화 기록 초기화 버튼
if st.sidebar.button("대화 기록 초기화"):
    st.session_state.chat_history = []
    st.rerun()