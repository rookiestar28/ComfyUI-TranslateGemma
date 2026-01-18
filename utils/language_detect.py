"""
Language auto-detection helpers for TranslateGemma.

TranslateGemma's official chat template requires an explicit `source_lang_code`.
The model card does not mention an "auto" source code, and the template will
raise if an unsupported code is provided. To keep the UX of "Auto Detect" in the
node UI, we perform best-effort local detection for text inputs.
"""

from __future__ import annotations

from typing import Optional

from .language_utils import LANGUAGE_MAP


_ZH_VARIANT_PAIRS: tuple[tuple[str, str], ...] = (
    # Common simplified/traditional character pairs (not exhaustive).
    # Format: (simplified, traditional)
    # High-frequency characters
    ("这", "這"),
    ("来", "來"),
    ("为", "為"),
    ("后", "後"),
    ("发", "發"),
    ("复", "復"),
    ("台", "臺"),
    ("里", "裡"),
    ("面", "麵"),
    ("国", "國"),
    ("汉", "漢"),
    ("语", "語"),
    ("书", "書"),
    ("马", "馬"),
    ("云", "雲"),
    ("网", "網"),
    ("单", "單"),
    ("东", "東"),
    ("门", "門"),
    ("开", "開"),
    ("关", "關"),
    ("见", "見"),
    ("车", "車"),
    ("长", "長"),
    ("风", "風"),
    ("鱼", "魚"),
    ("鸟", "鳥"),
    ("龙", "龍"),
    ("万", "萬"),
    ("与", "與"),
    ("体", "體"),
    ("画", "畫"),
    ("广", "廣"),
    ("极", "極"),
    ("线", "線"),
    ("爱", "愛"),
    ("学", "學"),
    ("写", "寫"),
    ("译", "譯"),
    ("际", "際"),
    ("间", "間"),
    ("过", "過"),
    ("进", "進"),
    ("当", "當"),
    ("时", "時"),
    ("问", "問"),
    ("难", "難"),
    ("应", "應"),
    ("经", "經"),
    ("现", "現"),
    ("术", "術"),
    ("业", "業"),
    ("统", "統"),
    ("动", "動"),
    ("传", "傳"),
    ("优", "優"),
    ("价", "價"),
    ("专", "專"),
    ("两", "兩"),
    ("决", "決"),
    ("别", "別"),
    ("删", "刪"),
    ("听", "聽"),
    ("员", "員"),
    ("图", "圖"),
    ("报", "報"),
    ("处", "處"),
    ("备", "備"),
    ("对", "對"),
    ("导", "導"),
    ("实", "實"),
    ("审", "審"),
    ("断", "斷"),
    ("换", "換"),
    ("显", "顯"),
    ("标", "標"),
    ("测", "測"),
    ("环", "環"),
    ("确", "確"),
    ("缩", "縮"),
    ("续", "續"),
    ("维", "維"),
    ("设", "設"),
    ("调", "調"),
    ("资", "資"),
    ("输", "輸"),
    ("错", "錯"),
    # Additional high-frequency pairs (TG-035 expansion)
    ("吗", "嗎"),
    ("湾", "灣"),
    ("历", "歷"),
    ("于", "於"),
    ("着", "著"),
    ("么", "麼"),
    ("还", "還"),
    ("没", "沒"),
    ("个", "個"),
    ("从", "從"),
    ("们", "們"),
    ("会", "會"),
    ("说", "說"),
    ("话", "話"),
    ("让", "讓"),
    ("请", "請"),
    ("头", "頭"),
    ("脑", "腦"),
    ("电", "電"),
    ("视", "視"),
    ("机", "機"),
    ("场", "場"),
    ("钱", "錢"),
    ("银", "銀"),
    ("办", "辦"),
    ("认", "認"),
    ("识", "識"),
    ("记", "記"),
    ("计", "計"),
    ("订", "訂"),
    ("论", "論"),
    ("议", "議"),
    ("词", "詞"),
    ("读", "讀"),
    ("课", "課"),
    ("讲", "講"),
    ("证", "證"),
    ("试", "試"),
    ("号", "號"),
    ("码", "碼"),
    ("总", "總"),
    ("种", "種"),
    ("类", "類"),
    ("组", "組"),
    ("织", "織"),
    ("纪", "紀"),
    ("约", "約"),
    ("红", "紅"),
    ("绿", "綠"),
    ("蓝", "藍"),
    ("黄", "黃"),
    ("亲", "親"),
    ("观", "觀"),
    ("规", "規"),
    ("觉", "覺"),
    ("乐", "樂"),
    ("欢", "歡"),
    ("医", "醫"),
    ("药", "藥"),
    ("厂", "廠"),
    ("厅", "廳"),
    ("县", "縣"),
    ("区", "區"),
    ("师", "師"),
    ("帅", "帥"),
    ("带", "帶"),
    ("杂", "雜"),
    ("虑", "慮"),
    ("惊", "驚"),
    ("战", "戰"),
    ("护", "護"),
    ("挥", "揮"),
    ("据", "據"),
    ("损", "損"),
    ("担", "擔"),
    ("择", "擇"),
    ("收", "收"),
    ("效", "效"),
    ("敌", "敵"),
    ("数", "數"),
    ("整", "整"),
    ("斗", "鬥"),
    ("旧", "舊"),
    ("无", "無"),
    ("旷", "曠"),
    ("权", "權"),
    ("杀", "殺"),
    ("条", "條"),
    ("构", "構"),
    ("柜", "櫃"),
    ("档", "檔"),
    ("楼", "樓"),
    ("欧", "歐"),
    ("步", "步"),
    ("残", "殘"),
    ("洁", "潔"),
    ("浅", "淺"),
    ("满", "滿"),
    ("演", "演"),
    ("灭", "滅"),
    ("灯", "燈"),
    ("热", "熱"),
    ("爷", "爺"),
    ("犹", "猶"),
    ("猎", "獵"),
    ("狮", "獅"),
    ("独", "獨"),
    ("疯", "瘋"),
    ("着", "著"),
    ("离", "離"),
    ("积", "積"),
    ("移", "移"),
    ("穷", "窮"),
    ("竞", "競"),
    ("策", "策"),
    ("简", "簡"),
    ("签", "簽"),
    ("纤", "纖"),
    ("绝", "絕"),
    ("给", "給"),
    ("细", "細"),
    ("继", "繼"),
    ("罗", "羅"),
    ("职", "職"),
    ("肃", "肅"),
    ("胜", "勝"),
    ("脏", "臟"),
    ("节", "節"),
    ("范", "範"),
    ("虚", "虛"),
    ("装", "裝"),
    ("补", "補"),
    ("规", "規"),
    ("览", "覽"),
    ("访", "訪"),
    ("许", "許"),
    ("评", "評"),
    ("诉", "訴"),
    ("详", "詳"),
    ("语", "語"),
    ("误", "誤"),
    ("调", "調"),
    ("谈", "談"),
    ("谢", "謝"),
    ("边", "邊"),
    ("运", "運"),
    ("选", "選"),
    ("递", "遞"),
    ("遗", "遺"),
    ("邮", "郵"),
    ("释", "釋"),
    ("钟", "鐘"),
    ("锁", "鎖"),
    ("键", "鍵"),
    ("镇", "鎮"),
    ("门", "門"),
    ("闻", "聞"),
    ("闭", "閉"),
    ("阅", "閱"),
    ("际", "際"),
    ("队", "隊"),
    ("险", "險"),
    ("随", "隨"),
    ("隐", "隱"),
    ("韩", "韓"),
    ("顾", "顧"),
    ("项", "項"),
    ("页", "頁"),
    ("预", "預"),
    ("飞", "飛"),
    ("马", "馬"),
    ("骗", "騙"),
    ("验", "驗"),
    ("惊", "驚"),
    ("黑", "黑"),
    # Additional expansion from web search (TG-035)
    ("买", "買"),
    ("卖", "賣"),
    ("华", "華"),
    ("级", "級"),
    ("军", "軍"),
    ("济", "濟"),
    ("兴", "興"),
    ("习", "習"),
    ("响", "響"),
    ("系", "係"),
    ("转", "轉"),
    ("质", "質"),
    ("争", "爭"),
    ("只", "隻"),
    ("众", "眾"),
    ("称", "稱"),
    ("创", "創"),
    ("树", "樹"),
    ("双", "雙"),
    ("参", "參"),
    ("虽", "雖"),
    ("岁", "歲"),
    ("阳", "陽"),
    ("艺", "藝"),
    ("亚", "亞"),
    ("游", "遊"),
    ("儿", "兒"),
    ("较", "較"),
    ("举", "舉"),
    ("净", "淨"),
    ("联", "聯"),
    ("连", "連"),
    ("练", "練"),
    ("临", "臨"),
    ("领", "領"),
    ("陆", "陸"),
    ("录", "錄"),
    ("虑", "慮"),
    ("卫", "衛"),
    ("为", "為"),
    ("务", "務"),
    ("协", "協"),
    ("携", "攜"),
    ("兄", "兄"),
    ("须", "須"),
    ("序", "序"),
    ("宣", "宣"),
    ("压", "壓"),
    ("严", "嚴"),
    ("养", "養"),
    ("样", "樣"),
    ("仪", "儀"),
    ("异", "異"),
    ("忆", "憶"),
    ("义", "義"),
    ("译", "譯"),
    ("阴", "陰"),
    ("营", "營"),
    ("影", "影"),
    ("拥", "擁"),
    ("涌", "湧"),
    ("优", "優"),
    ("忧", "憂"),
    ("余", "餘"),
    ("与", "與"),
    ("欲", "慾"),
    ("远", "遠"),
    ("园", "園"),
    ("约", "約"),
    ("跃", "躍"),
    ("杂", "雜"),
    ("糟", "糟"),
    ("责", "責"),
    ("窄", "窄"),
    ("张", "張"),
    ("涨", "漲"),
    ("赵", "趙"),
    ("折", "折"),
    ("针", "針"),
    ("镇", "鎮"),
    ("征", "徵"),
    ("证", "證"),
    ("郑", "鄭"),
    ("织", "織"),
    ("纸", "紙"),
    ("指", "指"),
    ("志", "志"),
    ("钟", "鐘"),
    ("终", "終"),
    ("种", "種"),
    ("周", "週"),
    ("朱", "朱"),
    ("烛", "燭"),
    ("筑", "築"),
    ("庄", "莊"),
    ("壮", "壯"),
    ("状", "狀"),
    ("准", "準"),
    ("资", "資"),
    ("综", "綜"),
    ("总", "總"),
    ("纵", "縱"),
    ("组", "組"),
    ("罪", "罪"),
    ("尊", "尊"),
    ("昨", "昨"),
    ("坐", "坐"),
    # More common pairs
    ("并", "並"),
    ("才", "才"),
    ("层", "層"),
    ("产", "產"),
    ("场", "場"),
    ("尝", "嘗"),
    ("车", "車"),
    ("陈", "陳"),
    ("成", "成"),
    ("诚", "誠"),
    ("程", "程"),
    ("迟", "遲"),
    ("冲", "衝"),
    ("虫", "蟲"),
    ("筹", "籌"),
    ("础", "礎"),
    ("处", "處"),
    ("传", "傳"),
    ("窗", "窗"),
    ("词", "詞"),
    ("辞", "辭"),
    ("村", "村"),
    ("达", "達"),
    ("带", "帶"),
    ("担", "擔"),
    ("党", "黨"),
    ("荡", "蕩"),
    ("导", "導"),
    ("灯", "燈"),
    ("敌", "敵"),
    ("递", "遞"),
    ("点", "點"),
    ("电", "電"),
    ("调", "調"),
    ("丢", "丟"),
    ("冻", "凍"),
    ("独", "獨"),
    ("读", "讀"),
    ("赌", "賭"),
    ("断", "斷"),
    ("队", "隊"),
    ("顿", "頓"),
    ("夺", "奪"),
    ("恶", "惡"),
    ("尔", "爾"),
    ("饭", "飯"),
    ("范", "範"),
    ("费", "費"),
    ("坟", "墳"),
    ("粉", "粉"),
    ("丰", "豐"),
    ("锋", "鋒"),
    ("凤", "鳳"),
    ("佛", "佛"),
    ("扶", "扶"),
    ("肤", "膚"),
    ("负", "負"),
    ("妇", "婦"),
    ("复", "復"),
    ("盖", "蓋"),
    ("干", "幹"),
    ("刚", "剛"),
    ("岗", "崗"),
    ("港", "港"),
    ("歌", "歌"),
    ("革", "革"),
    ("格", "格"),
    ("个", "個"),
    ("给", "給"),
    ("根", "根"),
    ("耕", "耕"),
    ("工", "工"),
    ("贡", "貢"),
    ("购", "購"),
    ("够", "夠"),
    ("构", "構"),
    ("顾", "顧"),
    ("刮", "刮"),
    ("挂", "掛"),
    ("观", "觀"),
    ("馆", "館"),
    ("惯", "慣"),
    ("广", "廣"),
    ("规", "規"),
    ("归", "歸"),
    ("轨", "軌"),
    ("贵", "貴"),
    ("柜", "櫃"),
    ("国", "國"),
    ("过", "過"),
    ("害", "害"),
    ("韩", "韓"),
    ("汉", "漢"),
    ("号", "號"),
    ("喝", "喝"),
    ("合", "合"),
    ("恨", "恨"),
    ("横", "橫"),
    ("红", "紅"),
    ("猴", "猴"),
    ("呼", "呼"),
    ("胡", "胡"),
    ("护", "護"),
    ("华", "華"),
    ("划", "劃"),
    ("化", "化"),
    ("画", "畫"),
    ("话", "話"),
    ("怀", "懷"),
    ("欢", "歡"),
    ("环", "環"),
    ("换", "換"),
    ("荒", "荒"),
    ("黄", "黃"),
    ("灰", "灰"),
    ("辉", "輝"),
    ("回", "回"),
    ("汇", "匯"),
    ("会", "會"),
    ("绘", "繪"),
    ("婚", "婚"),
    ("混", "混"),
    ("货", "貨"),
    ("获", "獲"),
    ("迹", "跡"),
    ("机", "機"),
    ("积", "積"),
    ("击", "擊"),
    ("鸡", "雞"),
    ("极", "極"),
    ("集", "集"),
    ("辑", "輯"),
    ("籍", "籍"),
    ("挤", "擠"),
    ("己", "己"),
    ("济", "濟"),
    ("继", "繼"),
    ("纪", "紀"),
    ("夹", "夾"),
    ("坚", "堅"),
    ("监", "監"),
    ("兼", "兼"),
    ("拣", "揀"),
    ("简", "簡"),
    ("剪", "剪"),
    ("见", "見"),
    ("件", "件"),
    ("建", "建"),
    ("健", "健"),
    ("舰", "艦"),
    ("剑", "劍"),
    ("将", "將"),
    ("奖", "獎"),
    ("讲", "講"),
    ("酱", "醬"),
    ("交", "交"),
    ("郊", "郊"),
    ("焦", "焦"),
    ("脚", "腳"),
    ("缴", "繳"),
    ("教", "教"),
    ("阶", "階"),
    ("结", "結"),
    ("杰", "傑"),
    ("节", "節"),
    ("洁", "潔"),
    ("姐", "姐"),
    ("解", "解"),
    ("届", "屆"),
    ("借", "借"),
    ("今", "今"),
    ("斤", "斤"),
    ("金", "金"),
    ("仅", "僅"),
    ("尽", "盡"),
    ("紧", "緊"),
    ("锦", "錦"),
    ("进", "進"),
    ("惊", "驚"),
    ("晶", "晶"),
    ("经", "經"),
    ("静", "靜"),
    ("镜", "鏡"),
    ("九", "九"),
    ("救", "救"),
    ("旧", "舊"),
    ("剧", "劇"),
    ("据", "據"),
    ("聚", "聚"),
    ("卷", "卷"),
    ("决", "決"),
    ("绝", "絕"),
    ("觉", "覺"),
)

_ZH_SIMPLIFIED_CHARS = {simp for simp, _trad in _ZH_VARIANT_PAIRS}
_ZH_TRADITIONAL_CHARS = {trad for _simp, trad in _ZH_VARIANT_PAIRS}


def _detect_zh_variant(text: str) -> Optional[str]:
    """
    Heuristic detection for Simplified vs Traditional Chinese.

    Returns:
        - "zh_Hant" if Traditional Chinese is more likely
        - "zh" if Simplified Chinese is more likely
        - None if undecidable
    """
    if not text:
        return None

    simplified = 0
    traditional = 0
    for ch in text:
        if ch in _ZH_SIMPLIFIED_CHARS:
            simplified += 1
        elif ch in _ZH_TRADITIONAL_CHARS:
            traditional += 1

    if simplified == 0 and traditional == 0:
        return None

    if traditional > simplified and traditional >= 2:
        return "zh_Hant"
    if simplified > traditional and simplified >= 2:
        return "zh"

    # Low-signal tie: do not guess.
    return None


def _detect_by_script(text: str) -> Optional[str]:
    # Japanese: Hiragana / Katakana
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF:
            return "ja"
        # Korean: Hangul syllables
        if 0xAC00 <= o <= 0xD7AF:
            return "ko"
        # Arabic
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            return "ar"
        # Hebrew
        if 0x0590 <= o <= 0x05FF:
            return "he"
        # Greek
        if 0x0370 <= o <= 0x03FF:
            return "el"
        # Thai
        if 0x0E00 <= o <= 0x0E7F:
            return "th"
        # Devanagari (Hindi/Marathi)
        if 0x0900 <= o <= 0x097F:
            return "hi"
        # Bengali
        if 0x0980 <= o <= 0x09FF:
            return "bn"
        # Gujarati
        if 0x0A80 <= o <= 0x0AFF:
            return "gu"
        # Gurmukhi (Punjabi)
        if 0x0A00 <= o <= 0x0A7F:
            return "pa"
        # Tamil
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
        # Telugu
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        # Kannada
        if 0x0C80 <= o <= 0x0CFF:
            return "kn"
        # Malayalam
        if 0x0D00 <= o <= 0x0D7F:
            return "ml"
        # Burmese (Myanmar)
        if 0x1000 <= o <= 0x109F:
            return "my"
        # Khmer
        if 0x1780 <= o <= 0x17FF:
            return "km"
        # Lao
        if 0x0E80 <= o <= 0x0EFF:
            return "lo"
        # Cyrillic (Russian/Ukrainian/Bulgarian/Serbian, etc.)
        if 0x0400 <= o <= 0x04FF:
            return "ru"
        # CJK ideographs (Chinese/Japanese)
        if 0x4E00 <= o <= 0x9FFF:
            # If we reached here, we did not see kana, so prefer Chinese.
            return "zh"

    return None


def _normalize_lang_code(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if not lang:
        return ""

    # Common aliases
    if lang == "iw":
        return "he"
    if lang == "in":
        return "id"
    if lang in ("fil", "tl"):
        return "tl"

    # Normalize separators / casing for regional codes.
    lang = lang.replace("-", "_")
    parts = lang.split("_", 1)
    if len(parts) == 2 and parts[1]:
        lang = f"{parts[0]}_{parts[1].upper()}"
    return lang


def detect_source_lang_code(text: str, fallback: str = "en") -> str:
    """
    Best-effort language detection for text inputs.

    Returns a code that is present in LANGUAGE_MAP when possible. If no reliable
    detection is available, returns `fallback` (default: "en").
    """
    if not text or not text.strip():
        return fallback

    supported = set(LANGUAGE_MAP.keys())

    by_script = _detect_by_script(text)
    if by_script and by_script in supported:
        if by_script == "zh":
            return _detect_zh_variant(text) or by_script
        return by_script

    # Optional statistical classifier (preferred when installed).
    # Keep dependency optional at runtime; requirements can include it for best UX.
    try:
        import langid  # type: ignore
    except Exception:
        langid = None

    if langid is not None:
        try:
            lang, _score = langid.classify(text)
            lang = _normalize_lang_code(lang)
            # Prefer explicit region mapping when classifier provides it.
            if lang.startswith("zh_"):
                region = lang.split("_", 1)[1]
                if region in {"TW", "HK", "MO"} and "zh_Hant" in supported:
                    return "zh_Hant"
                if region in {"CN", "SG", "MY"} and "zh" in supported:
                    return "zh"
            if lang in supported:
                return lang
            # Try base language for regional variants (e.g., pt_BR -> pt).
            base = lang.split("_", 1)[0]
            if base in supported:
                if base == "zh":
                    return _detect_zh_variant(text) or base
                return base
        except Exception:
            pass

    return fallback
