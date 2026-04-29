# Configuration for car accident prediction project

COLUMN_MAPPING = {
    "ZINSRDAVL": "연령대",
    "ZIOSEXCD": "성별",
    "ZDPRODSCD": "국산차량여부",
    "NCR": "직전3년간사고건수",
    "ZCARPSGVL": "차량경과년수",
    "ZDRVLISCD___T": "운전자한정특별약관",
    "ZENTCARCD": "가입경력코드",
    "ZCARISDAM": "차량가입금액",
    "ZIMAGERVL": "영상기록장치특약가입",
    "YUHO": "유효대수",
    "SAGO": "사고건수"
}

DROP_COLUMNS = ['ZCPRLCLCD', 'ZDRVLISCD']

# Category mappings for manual label encoding
ACCIDENT_HISTORY_MAP = {'Z': 0, 'N': 1, 'D': 2, 'C': 3, 'B': 4}
CAR_AGE_MAP = {'신차': 0, '5년이하': 1, '10년이하': 2, '10년이상': 3}
CAR_TYPE_MAP = {
    '소형A': 0, '소형B': 1, '중형': 2, '대형': 3,
    '다목적1종': 4, '다목적2종': 5, '기타': 6
}
INSURANCE_AMOUNT_MAP = {'미가입': 0, '5천만원이하': 1, '1억이하': 2, '1억이상': 3}
MILEAGE_MAP = {
    '3000K': 0, '5000K': 1, '7000K': 2, '10000K': 3,
    '12000K': 4, '15000K': 5, '미가입': 6
}
