import cv2 as cv
import io
from PIL import Image
import streamlit as st
import streamlit_extras.switch_page_button as ste
import tensorflow as tf
import numpy as np
import pytesseract
import string
from pdf2image import convert_from_path, convert_from_bytes
from streamlit_pdf_viewer import pdf_viewer
from typing import Dict
from gtts import gTTS
import webbrowser
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, PageBreak, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
api_key = "c2927870-d989-11ee-a222-bf997f2bfeae"
user_id = "188821"
st.set_page_config(layout="wide")
if ("login" in st.session_state) and (st.session_state.login):
	pass
else:
	ste.switch_page("Login")
VOCABS: Dict[str, str] = {
    "digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "punctuation": string.punctuation,
    "currency": "£€¥¢฿",
    "ancient_greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    "arabic_letters": "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي",
    "persian_letters": "پچڢڤگ",
    "hindi_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_punctuation": "؟؛«»—",
    "sanskrit_unicode": "ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓क़ख़ग़ज़ड़ढ़फ़य़ॠॢॣ।॥०१२३४५६७८९॰ॱॲॳॴॵॶॷॸॹॺॻॼॽॾॿ",
    "sanskrit_numerals": "१२३४५६७८९०",
}

VOCABS['bengali'] = 'শ০ূৃৰজআঔঅঊিঢ়খ৵পঢই৳ফঽ৪লেঐযঃঈঠুধড়৲ৄথভটঁঋৱরডৢছ৴ঙওঘস১৹ণগ৷৩ত৮হ৭োষৎ৶কন৬চমৈা়ীৠঝএ৻ব৯য়উৌঞ৺২ংৣদ৫্ৗ-।'
VOCABS['gujarati'] = '૮લઔ૨સખાઑઈઋૐઓવૄ૦઼ઁનઞઊ૫ીશફણ૬૭બ૧રળૌુઠઐઉષપેઇઅૃઝજૉક૱૯ગઍદો૪ૅએંહડઘ૩ૂછઙઃઽટતધિૈયઢ્આમથચભ-'
VOCABS['gurumukhi'] = 'ਗ਼ਵਨਁਰਊਖਂਆਜੈਲੴਣ੧ਛਭਫ੮੯ਚਔੀਯਹਲ਼ਞ੩ੜਫ਼ੁਮ੫ਤੇਦਸ਼ਟੰ੭ਓਅਃਡਾਉਠੱਈ੦ੵਖ਼ਏਕਥ੬ਧੲੑਝਿ੨ਐਬਪਘਸ਼ਙੌਜ਼ੋਗ੍ੳਇ੪ੂਢ-।'
VOCABS['kannada'] = 'ಚೕಒಉೖಂಲಾಝಟೆಅ೬ೇ೨ಬಡವಜಢಞಔಏಧಶಭತಳೀಕಐಈಠಪ೫ಣ೮ೞಆಯುಗೢಋದಘೂ್ೈ೦ಓಱಃಹ೯ೋಮ೭ೠಥಖಫಇರ೪ಛಙೣಿ೩ೌೄಷಌಸನ಼ಊಎ೧ೃೊ-'
VOCABS['malayalam'] = '൪ഉ൮ള൵ഔംസഞഎഷ൫ൄൌ-ഃൈീഌഛഇണാഈഹധ൭ജച൱൴൹യതൻശഒ൯ഗർഊആവഖൠൣ൩ോൽ൧അ൳ൗപഭൃ്മെഐൡഓദഏറിഠരൺ൰ൾങട൦ഢൢഡലേഴഝൊ൲ബനൂഥൿഘഫുഋക൬൨ '
VOCABS['odia'] = 'ଖ୯୬ୋଓଞ୍ଶ୪ଣଥଚରୄତଃେ୮ଆକଵୂନଦ୰ୖୢଜଉଳଅଁଲଯଔପ୭ଷଢଡ଼ଊୟମିୁ୧ଂ଼ୀବଟଭଢ଼୦ଘଠୗ୫ୡାଐ୨ଙହଈୱ୩ୃଛଏୌଗଫସଇଧଡଝୈୣୠଋ-।'
VOCABS['tamil'] = 'ய௴ஷ௫ைெஸஎஈோவ௲ூு௭அ்ஶி௰ஹ௧ௐா௮ஔ௺சீண௩இனஆழ௪௯ஙஊதஜ௷௶மௌள௸ஐபநேற௬டஒ௹ஞஉஏகௗொர௱௵ஃ௨லஓ௳௦-'
VOCABS['telugu'] = '౦ఱకఆఋడత౯౻ిహౌ౭౽ఉ౮్ధఓగ౼మ౫ూౠఔాఇనైఁజీౄుేసశృఃఝఢరఠలోఞౘఅ౹౧ౢఛబ౸ఐయ౩ఖటచెొఊదఈషథభఏౙ౬౾ఎ౪ణఒప౨ఫంఘఙళవ౺-'
VOCABS['urdu'] = 'ٱيأۃدےش‘زعكئںسحٰنؐةقذ؟ؔ۔—ًمھٗپغٖطإؒرڑصټٍگاؤجضْﷺچ‎ۓِّؓٹظىتڈ‍یُه،خو؛آفبؑلہثﺅ‌ژَۂءک‏'


VOCABS['hindi'] = 'ॲऽऐथफएऎह८॥ॉम९ुँ१ं।षघठर॓ॼड़गछिॱटऩॄऑवल५ढ़य़अञसऔयण॑क़॒ौॽशऍ॰ूीऒॊख़उज़ॻॅ३ओऌळनॠ०ेढङ४़ॢग़पऊॐज२डैभझकआदबऋखॾ॔ोइ्धतफ़ईृःा६चऱऴ७-'
VOCABS['sanskrit']='ज़ऋुड़ऍऐक५टय४उः३ॠध९्७ू१वऌौॐॡॢइ६ाै८नृअंथढेखऔघग़०लजोईरञपफँझभषॅॄगतचहसीढ़आशए।म२दठङबिऊडओळछण़ऽ'
VOCABS['devanagari'] = 'रचख़३ॾऍृेञलॻॉऴषॐॢ१य०ॽएा२ई।ग़७टऐय़॥तोदऽभुनओऒ-ठँ.ौ्८ॼझॠविःक़ी॰छॅॊऩऱ़थजशळङअऋखबफउ५फ़६ऊॲॆज़कढ़मूस॓इऔह॑ैगढॣधआड़९ं४डणपॄघऑ'

def create_pdf(pages_content):
    styles = getSampleStyleSheet()

    pdfmetrics.registerFont(TTFont('Regular', 'GoNotoKurrent-Regular.ttf'))
    styles['Normal'].fontName = 'Regular'
    styles['Normal'].fontSize = 15
    styles['Normal'].leading = 16

    content = []

    for page_content in pages_content:
        for paragraph_text in page_content:
            paragraph = Paragraph(paragraph_text, styles["Normal"])
            content.append(paragraph)
            content.append(Paragraph("<br/><br/>", styles["Normal"]))  # Add some space between paragraphs
        content.append(PageBreak())  # Add space between pages

    return content

def get_word(img):
  box_count = 0
  img1 = cv.cvtColor(np.array(img),cv.COLOR_BGR2RGB)
  img2 = img1.copy()
  #converting to grayscale
  gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
  #applying gaussian blur
  blur = cv.GaussianBlur(gray,(11,11),0)
  #thresholding to convert into binary
  thresholded = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,159,3)
  #dilate image to connect text contours
  kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,3))
  dilated = cv.morphologyEx(thresholded,cv.MORPH_CLOSE,kernel)
  # dilated = cv.dilate(thresholded,kernel,iterations=1)
  #get external contours
  contours = cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours)==2 else contours[1]
  #draw contours
  for points in contours:
    pad = 10
    x,y,w,h = cv.boundingRect(points)
    box = img1[y:y+h,x:x+w,:]
    # print(list(box.flatten()).count(255),box.shape)
    box_count+=1
    temp_img = Image.fromarray(box)
    # temp_img.save(str("/content/drive/MyDrive/Project/Books/Words/Kannada words/img_"+str(box_count+total_boxes)+".jpg"))
    # print("/content/drive/MyDrive/Project/Books/Words/Kannada words/img_"+str(box_count+total_boxes)+".jpg")
    cv.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
  # plt.figure(figsize=(10,10))
  # plt.imshow(img1)
  # plt.axis('off')
    if(w>=100 and h>=20):
      return temp_img
    # print(temp_img.size)
      # plt.imshow(temp_img)
    # plt.axis('off')
  # return box
def results(pages,lang):
  pages_content = []
  for i in range(len(pages)):
      data = pytesseract.image_to_data(pages[i], output_type=pytesseract.Output.DICT,lang=lang)
      block_text = {}
      for i in range(len(data['block_num'])):
        block_num = data['block_num'][i]
        text = data['text'][i].strip()
        if block_num not in block_text:
          block_text[block_num] = []
        if text:
          block_text[block_num].append(text)
      block_texts = []
      for block_num, text_list in block_text.items():
        if(block_num==0):
          continue
        temp_text = ''
        for text in text_list:
          temp_text+=text
          temp_text+=' '
        block_texts.append(temp_text)
      pages_content.append(block_texts)
  return pages_content
  st.write(pages_content)
  page_data =  pytesseract.image_to_data(Image.open(img),lang=lang,output_type=pytesseract.Output.DICT)
  extracted_text = ' '.join([word for word in page_data['text'] if word.strip()])
  return extracted_text
def initialize_handwritten_models(lang):
  models_dict = {"telugu":"Models/crnn_vgg16_bn_telugu.pt","hindi":"Models/crnn_vgg16_bn_hindi.pt"}
  det_model = db_resnet50(pretrained=True)
  reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False,vocab=VOCABS[lang])
  reco_dict = torch.load(models_dict[lang],map_location="cpu")
  reco_model.load_state_dict(reco_dict)

  det_predictor = DetectionPredictor(
    PreProcessor(
        (1024, 1024),
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287)
    ),
    det_model
  )

  reco_predictor = RecognitionPredictor(
    PreProcessor(
        (32, 128),
        preserve_aspect_ratio=True,
        batch_size=32,
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301)
    ),
    reco_model
  )
  predictor = OCRPredictor(det_predictor, reco_predictor)
  return predictor
def get_txt(json_op,img,lang):
  result=''
  for page in json_op['pages']:
    for block in page['blocks']:
      for lines in block['lines']:
        for word in lines['words']:
          result+=word['value']+' '
          result+='\n'
  return results(img,lang)
def predict(img,lang):
  output = results(img,lang)
  #doc = DocumentFile.from_images(img)
  #img = Image.open(img)
  #model=initialize_handwritten_models(lang)
  #result = model(doc)
  #output = get_txt(result.export(),img,lang)
  return output
if ("login" in st.session_state) and (st.session_state.login):
	pass
else:
	ste.switch_page("Login")
col1,col2,col3 = st.columns([1,3,1])
with col2:
	st.title("Integrated Multilingual Image Processing")
st.divider()
col1,col2,col3 = st.columns([1,1,1])
with col2:
	logo = Image.open("logo.png")
	st.image(logo,width=400)
col1,col2,col3 = st.columns([2,4,1])
col1,col2,col3 = st.columns([1,10,1])
with col2:
	st.markdown("###### An integrated tool to effortlessly understand and communicate across languages through image recognition, OCR, translation, and text-to-speech.")
st.divider()
st.header("Upload PDF File:")
pdf = st.file_uploader("",type='pdf')
op = {0:'English',1:'Gujarati', 2:'Hindi', 3:'Kannada',4:'Telugu'}
lang_code = {0:"eng",1:"guj",2:"hin",3:"kan",4:"tel"}
speech_codes_1 = {"hindi":"hi","gujarati":"gu","marathi":"mr","kannada":"kn"}
speech_codes_2 = {"english":"en","french":"fr","portuguese":"pt","spanish":"es"}
translator = GoogleTranslator(source='auto')
lang_codes = translator.get_supported_languages(as_dict=True)
extract_text = False
translate_text = False
to_speech = False
if(pdf is not None):
  temp_pdf = pdf
  # pdf_viewer(temp_pdf.read())
  images = convert_from_bytes(pdf.read(), 250)
  pdf.seek(0)
  col1,col2 = st.columns([1,1])
  with col1:
      pdf_viewer(pdf.read(),height=1000)
  pages = []
	# st.write(enumerate(images))
  for i, image in enumerate(images):
      pages.append(image)
  word = get_word(pages[0])
  # st.image(word)
  img = word.resize((224,224))
	#img = tf.keras.utils.load_img(word,target_size=(224,224,3))
  img1 = tf.keras.preprocessing.image.img_to_array(img)
  img1 = np.expand_dims(img,axis=0)
  model = tf.keras.models.load_model("alexnet_5.h5",compile=False)
  model.compile(optimizer='adam',loss='categorical_crossentropy')
  pred = model.predict(img1)
  label = np.argmax(pred)
  # detected_lang = op[label]
  text = predict(pages,lang_code[label])
  with col2:
    target_lang = st.selectbox("Select the target language for translation",list(lang_codes.keys()),index=45)
    translated_text = []
    for i in text:
      translated_text_temp = []
    # st.write(i)
      for j in i:
        # st.write(j)
        translated = GoogleTranslator(source='auto', target=target_lang).translate(j)
        # st.write(translated)
        translated_text_temp.append(translated)
      translated_text.append(translated_text_temp)
  # st.write(translated_text)
    content = create_pdf(translated_text)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(content)
    buffer.seek(0)
    translated_pdf: bytes = buffer.getvalue()
  # translated_pdf = doc.build(content)
    pdf_viewer(translated_pdf,height=900)
  st.download_button("Download PDF",data=translated_pdf,file_name='translated_pdf.pdf')
