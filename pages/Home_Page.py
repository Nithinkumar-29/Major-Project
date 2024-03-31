import cv2 as cv
from PIL import Image
import streamlit as st
import streamlit_extras.switch_page_button as ste
import tensorflow as tf
import numpy as np
import pytesseract
import string
from typing import Dict
from gtts import gTTS
import webbrowser
from deep_translator import GoogleTranslator
from streamlit.components.v1 import html
def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
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
def get_word(img):
  box_count = 0
  img1 = cv.cvtColor(np.array(img),cv.COLOR_BGR2RGB)
  img2 = img1.copy()
  #converting to grayscale
  gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
  #applying gaussian blur
  blur = cv.GaussianBlur(gray,(9,9),0)
  #thresholding to convert into binary
  thresholded = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,159,3)
  #dilate image to connect text contours
  kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,1))
  dilated = cv.morphologyEx(thresholded,cv.MORPH_CLOSE,kernel)
  # dilated = cv.dilate(thresholded,kernel,iterations=1)
  #get external contours
  contours = cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours)==2 else contours[1]
  #draw contours
  for points in contours:
    pad = 10
    x,y,w,h = cv.boundingRect(points)
    tempp = np.array(img)
    box = tempp[y:y+h,x:x+w,:]
    # print(list(box.flatten()).count(255),box.shape)
    box_count+=1
    temp_img = Image.fromarray(box)
    if(w>=100 and h>=20):
      return temp_img
    # print(temp_img.size)
      # plt.imshow(temp_img)
    # plt.axis('off')
  # return box
def results(img,lang):
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
st.header("Upload Image for Analysis:")
img = st.file_uploader("")
op = {0:'English',1:'Gujarati', 2:'Hindi', 3:'Kannada',4:'Telugu'}
lang_code = {0:"eng",1:"guj",2:"hin",3:"kan",4:"tel"}
speech_codes_1 = {"hindi":"hi","gujarati":"gu","marathi":"mr","kannada":"kn"}
speech_codes_2 = {"english":"en","french":"fr","portuguese":"pt","spanish":"es"}
translator = GoogleTranslator(source='auto')
lang_codes = translator.get_supported_languages(as_dict=True)
extract_text = False
translate_text = False
to_speech = False
if(img is not None):
	temp_img = img
	image = Image.open(img)
	word = get_word(image)
	st.write(word.size)
	word = word.convert("RGB")
	img2 = word.resize((224, 224))
	st.write(img2.size)
	st.image(img2)
	#img = tf.keras.utils.load_img(word,target_size=(224,224,3))
	#img1 = tf.keras.preprocessing.image.img_to_array(img2)
	#st.write(img1.shape)
	img1 = np.expand_dims(img2,axis=0)
	st.write(img1.shape)
	model = tf.keras.models.load_model("alexnet_5.h5",compile=False)
	model.compile(optimizer='adam',loss='categorical_crossentropy')
	pred = model.predict(img1)
	label = np.argmax(pred)
	col1,col2,col3 = st.columns([1,3,1])
	with col2:
		st.image(image,width=1000)
		st.markdown("##### Detected language: "+op[label])
		extract_text = st.button("Extract text",key="text")
		if extract_text:
			st.session_state.button1 = True
	col1,col2,col3 = st.columns([1,1,1])
	if st.session_state.button1:
		with col1:
			text = predict(temp_img,lang_code[label])
			st.text_area(label="Extracted text",value=text,height=500)
			translate_text = st.button("Translate text",key="translate")
			if translate_text:
				st.session_state.button2 = True
			if st.session_state.button2:
				with col2:
					target_lang = st.selectbox("Select the target language for translation",list(lang_codes.keys()))	
					translated = GoogleTranslator(source='auto', target=lang_codes[target_lang]).translate(text)
					st.text_area(label="Translated text",value=translated,height=400)
					to_speech = st.button("Generate speech",key="generate")
					if to_speech:
						st.session_state.button3 = True
					if st.session_state.button3:
						with col3:
							if target_lang in list(speech_codes_2.keys()):
								speech = gTTS(text=translated,lang=speech_codes_2[target_lang],slow=False)
								speech.save("sample.mp3")
								st.write("Generated speech")
								st.audio("sample.mp3")
							elif target_lang in list(speech_codes_1.keys()):
								url = "https://ivrapi.indiantts.in/tts?type=indiantts&text="+translated+"&api_key="+api_key+"&user_id="+user_id+"&action=play&numeric=hcurrency&lang="+speech_codes_1[target_lang]+"_female_v1&ver=2"
								st.write("Generated speech")
								temp_bool = st.button('Play Audio', on_click=open_page, args=(url,))
							else:
								st.write("Can only generate speech to one of the languages:\n1. English\n2. French\n3. Portuguese\n4. Spanish\n5. Hindi\n6. Marathi\n7. Gujarati\n8. Kannada ")
col1,col2,col3 = st.columns([8,1,8])
with col2:
	st.markdown("[About Us](AboutUs)")
