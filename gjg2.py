__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
    layout="wide"
)

# --------------------------------->
# 데이터 로드 및 전처리
# --------------------------------->
df = pd.read_csv("광진구_추천업소_최종데이터333.csv")
# 위도, 경도 결측치 제거
df = df.dropna(subset=['latitude', 'longitude'])

# --------------------------------->
# ChromaDB 관련 함수 정의
# --------------------------------->
# ChromaDB 클라이언트 초기화 (캐싱 적용)
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="chroma_db3")

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
st.title("광진구 착한가격 업소 챗봇")
st.write("착한가격 업소 데이터에 대해 질문해보세요.")


# --------------------------------->
# 사이드바 설정 UI
# --------------------------------->
with st.sidebar:
    # 이미지 표시
    try:
        image = Image.open("sub0402_img01.png")
        st.image(image)
    except FileNotFoundError:
        st.warning("사이드바 이미지를 찾을 수 없습니다.")   

    # 컬렉션션 섹션
    collection_name = "gjg_report"
    # 사용 가능한 컬렉션을 불러오되, 'gjg_report'만 남기기
    all_cols = get_available_collections()
    allowed = [c for c in all_cols if c == "gjg_report"]

    # 설정 섹션
    st.sidebar.markdown("---")    
    api_key = st.text_input("OpenAI API 키를 입력하세요", type="password")

    # 필터링 섹션
    st.sidebar.markdown("---")
    shop_search = st.text_input("🔍 사업장명 검색")      

    # 지도 필터링 UI   
    st.sidebar.subheader("지도 필터")
    selected_category = st.sidebar.multiselect("업종 선택", df['업종'].unique())
    상품권_선택 = st.sidebar.selectbox("서울사랑상품권 가맹", ['전체', '가능', '불가능'])

# --------------------------------->
# 지도 필터링 로직
# --------------------------------->
filtered_df = df.copy()
filters_applied = False  # 필터가 적용되었는지 여부를 추적하는 변수

# 검색 필터링
if shop_search:
    # 대소문자 구분 없이 포함 매칭
    filtered_df = filtered_df[filtered_df['사업장명'].str.contains(shop_search, case=False, na=False)]
    filters_applied = True

# 카테고리 필터링
if selected_category:
    filtered_df = filtered_df[filtered_df['업종'].isin(selected_category)]
    filters_applied = True

# 상품권 필터링
if 상품권_선택 != '전체':
    filtered_df = filtered_df[filtered_df['사랑상품권'] == 상품권_선택]
    filters_applied = True

# 필터링된 결과가 있을 경우 테이블 표시
if filters_applied and not filtered_df.empty:
    st.markdown("### 필터링된 업소 정보")
    # 보여주고 싶은 컬럼만 선택
    cols_to_show = ['사업장명', '업종', '사랑상품권', '도로명주소']
    # 대화형 테이블 (스크롤·검색 가능)
    st.dataframe(
        filtered_df[cols_to_show].reset_index(drop=True), 
        use_container_width=True)
elif filters_applied and filtered_df.empty:
    # Streamlit에서 같은 메시지가 중복 출력되지 않도록 경고 메시지 조건을 분리
    st.session_state['no_results_shown'] = st.session_state.get('no_results_shown', False)

    if not st.session_state['no_results_shown']:
        st.warning("조건에 맞는 업소가 없습니다. 필터를 조정해주세요.")
        st.session_state['no_results_shown'] = True

# --------------------------------->
# 지도 시각화 UI
# --------------------------------->
# 지도 섹션 제목
st.header("지도로 보기")

# 필터링된 데이터 여부에 따른 설명
if filters_applied:
    if not filtered_df.empty:
        st.success(f"총 {len(filtered_df)}개의 업소가 필터링되었습니다.")
    else:
        st.warning("조건에 맞는 업소가 없습니다. 필터를 조정해주세요.")
else:
    st.info("필터가 적용되지 않았습니다.")

# 지도 중심점 설정
center = [37.5502596, 127.073139]
m = folium.Map(location=center, zoom_start=15)

# 업종별 색상 정의
category_colors = {
    "일반음식점": "darkred",
    "휴게음식점": "lightred",
    "세탁업": "blue",
    "목욕장업": "darkblue",
    "숙박업": "cadetblue",
    "미용업": "pink",
    "이용업": "purple",
    "안경업": "orange"        
}


# 필터가 선택되지 않은 경우 광진구 중심 표시
if not filters_applied:
    folium.Marker(
        location=center,
        popup="광진구 중심",
        tooltip="광진구",
        icon=folium.Icon(color="red", icon="glyphicon-map-marker")
    ).add_to(m)
    
# 필터링된 결과가 있는 경우: 업소 마커 추가
elif not filtered_df.empty:
    for idx, row in filtered_df.iterrows():
        업종 = row['업종']
        color = category_colors.get(업종, "blue")
        
        html = f"""
<div style="font-family: 'Arial'; border-radius: 8px; padding: 10px; background-color: #f9f9f9;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1); display: flex; flex-direction: column; justify-content: flex-start;">
    <h4 style="margin: 0 auto 8px auto; color: #2c3e50; text-align: center;">{row['사업장명']}</h4>
    <p style="margin: 3px 0;"><b>업종:</b> {row['업종']}</p>
    <p style="margin: 3px 0;"><b>유형:</b> {row['업태구분명']}</p>
    <p style="margin: 3px 0;"><b>평균가격:</b> {row['가격']}</p>
    <p style="margin: 3px 0;"><b>상품권 가맹:</b> {row['사랑상품권']}</p>
    <p style="margin: 3px 0;"><b>위생등급:</b> {row['위생등급'] if pd.notna(row['위생등급']) else '미지정'}</p>
    <p style="margin: 3px 0;"><b>행정처분:</b> {row['행정처분'] if pd.notna(row['행정처분']) else '없음'}</p>
</div>

"""
        iframe = IFrame(html, width=300, height=240)
        popup = folium.Popup(iframe, max_width=300)
        tooltip = row['사업장명']
        marker = folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            tooltip=tooltip,
            icon=folium.Icon(color=color, icon="glyphicon-search")
        )
        marker.add_to(m)
    
    # 필터링된 결과가 있으면 마커 범위로 지도 조정
    if len(filtered_df) > 0:
        bounds = [
            [filtered_df['latitude'].min(), filtered_df['longitude'].min()],
            [filtered_df['latitude'].max(), filtered_df['longitude'].max()]
        ]
        m.fit_bounds(bounds)


# 지도 컨트롤 추가
folium.LayerControl().add_to(m)

# 지도 표시
st_folium(m, width="100%", height=600)



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

        system_prompt = """당신은 광진구 착한가격업소 심사 전문 평가 AI입니다. 광진구청 또는 행정안전부의 착한가격업소 지정 기준에 따라 업소 후보 리스트를 평가하고, 지정 요건 충족 여부를 심사합니다.

        다음의 평가 기준을 반드시 반영해 업소별 적격성 여부를 판단하고 설명하세요:

        답변 작성 가이드라인:
        1. 질문과 직접 관련된 정보만 명확하게 전달하세요.
        2. 문서에서 확인되지 않거나 추정에 기반한 내용은 포함하지 마세요.
        3. 광진구 또는 행정안전부의 착한가격업소 기준에 기반한 정보만 사용하세요.
        4. 질문이 업소 평가와 관련된 경우, 지정 필수 조건과 우대 사항을 기준으로 판단하고 그 이유를 명시하세요.
        5. 각 업소의 업종이 요식업인지 비요식업인지에 따라 아래 평가표 기준에 따라 점수를 계산하고, 총점이 40점 이상이면 "지정 가능", 미만이면 "지정 불가"로 분류하세요.

        - 필수 결격 사유(가격 초과, 행정처분, 세금 체납 등)가 있을 경우, 점수와 관계없이 "지정 불가"로 표시하세요.
        - 각 항목별 점수와 총점을 명시하고, 지정 가능 여부 및 간단한 사유를 함께 출력하세요.


        [분석 기준]
        1. 해당 업종 평균 대비 저렴한 가격
        2. 위생 상태 및 청결 기준 충족
        4. 지역 화폐 가맹점 여부, 지역 공헌 활동 참여
        5. 가격 동결·인하 등 물가 안정 기여 노력

        [추가 선별 분석 기준]
        1. 주요 품목 가격이 지역 평균보다 저렴한가?
        2. 옥외 가격표시제 이행 여부
        3. 원산지 표시제 이행 여부 (음식업종)
        4. 최근 2년 내 행정처분 이력 없음
        5. 최근 1년 내 휴업 사실 없음
        6. 지방세 3회 이상 또는 100만 원 이상 체납 없음

        [우대 사항]
        1. 지역화폐 가맹 여부
        2. 식품위생등급제 신청 여부 (식품업종 위생 평가)
        3. 지역사회 공헌 활동 (예: 경로우대, 청소년 할인, 특정시간 할인)
        4. 최근 5년간 표창 수상 여부 (시장·구청장 이상)


        [요식업 평가표 기준 – 총점 55점]

        1. 가격 (최대 30점)
        - 가격수준 (30점): ‘착한가격메뉴’가 인근 상권 평균보다 저렴하면 30점 (신규 지정 기준)
        - 가격안정 노력 (10점): 가격 동결 유지 기간에 따라 재지정 시만 평가
            • 2년 이상 유지: 10점
            • 1년 이상 2년 미만: 5점
            ※ 신규지정 시에는 미평가

        2. 위생·청결 (최대 20점)
        - 아래 조건 중 하나 충족 시 20점 간주
            • 식품의약품안전처 위생등급제 ‘좋음’ 이상
            • 최근 1년 내 위생 지도·점검 결과 우수

        또는 아래 세부 평가로 구성:
            • 주방 (10점)
                - 바닥, 배수시설 청결(5점) + 위생복·소독도구 사용(5점)
            • 매장 (7점)
                - 식탁 정리, 소독용품 비치, 정수기 관리 등
            • 화장실 (3점)
                - 세척제·위생종이 등 위생설비 구비, 환기 등
                ※ 매장 단독 화장실일 경우만 개별 평가

        3. 공공성 (최대 5점) – 항목당 1점
            • 고아원 양로원 기부
            • 착한가격업소 표찰 부착
            • 지역화폐 가맹
            • 지역특화자원 활용 (로컬푸드 등)
            • 기타 지역사회 공헌도

        [비요식업 평가표 기준 – 총점 55점]

        1. 가격 (최대 30점)
        - 가격수준 (30점): 인근 상권 평균보다 저렴하면 30점 (신규 지정 기준)
        - 가격안정 노력 (10점): 재지정 시만 평가
            • 3년 이상 유지: 10점
            • 1년 이상 3년 미만: 5점
            ※ 신규지정 시에는 미평가

        2. 위생·청결 (최대 20점)
        - 최근 1년 내 관련 법령 기반 지도·점검 결과 우수 시 20점 간주
        또는:
            • 주영업시설 (15점): 시설 및 용품의 위생·청결
            • 화장실 (5점): 세척제·소독설비·위생종이 등 구비 여부
                ※ 단독 화장실일 경우만 개별 평가

        3. 지역사회기부와 공헌도 (최대 5점) – 항목당 1점
            • 고아원 양로원 기부
            • 기타 지역사회 공헌도
            • 지역화폐 가맹점점
            • 착한가격업소 표찰 부착착
            • 지역특화자원 활용도


        

        """

        user_prompt = f"""{context}

        사용자 질문: {query}

        위 문서들을 바탕으로 사용자 질문에만 정확히 답변해주세요.
        광진구 착한가격업소 지정 기준인 가격, 위생, 서비스, 공공성 등 주요 분석 기준을 포함하여 분석하고, 신규 후보 업소를 발굴해주세요.
        각 업소의 업종이 요식업인지 비요식업인지에 따라 아래 평가표 기준에 따라 점수를 계산하고, 총점이 40점 이상이면 "지정 가능", 미만이면 "지정 불가"로 분류하세요.
        프렌차이즈 업소는 기준에 부적합하므로 평가에서 제외 해주세요.
        제공된 광진구 업소 목록을 정리해주세요. 
        특히 대전의 성심당처럼 사회공헌 활동을 활발히 하거나, 
        주변 지역에 실질적인 도움을 주는 지역 기여도가 높은 업소가 있다면 이를 고려하여 점수를 매기고 함께 표시해주시면 좋겠습니다.

        - 필수 결격 사유(가격 초과, 행정처분, 세금 체납 등)가 있을 경우, 점수와 관계없이 "지정 불가"로 표시하세요.
        - 각 항목별 점수와 총점을 명시하고, 지정 가능 여부 및 간단한 사유를 함께 출력하세요.
        - 마지막란에 리뷰도 불러와주세요

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

#  # 컬렉션 정보 표시
# if collection:
#     try:
#         count = collection.count()
#         st.sidebar.success(f"컬렉션 '{collection_name}'에서 {count}개의 문서를 불러왔습니다.")
#     except Exception as e:
#         st.sidebar.sidebar.warning(f"컬렉션 정보 확인 중 오류: {e}")
# else:
#     st.warning(f"컬렉션을 선택하거나 찾을 수 없습니다. 컬렉션 목록을 확인하세요.")

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
st.sidebar.markdown("---")
st.sidebar.header("예시 질문")
example_questions = [
    "광진구에서 착한가격업소는 어떤 기준으로 선정되나요?",
    "혼밥하기 좋은 식당 추천해줘",
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

