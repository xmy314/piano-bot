// Libraries
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <AccelStepper.h>

// Physical Bot Characteristics:
#define LEFT_LIMIT_SWTICH_RELATIVE_TO_A0 196
#define RIGHT_LIMIT_SWTICH_RELATIVE_TO_A0 1544
#define STEPPER_TICK_PER_KEY 2.3 * 1000 / 61.5
#define LEFT_KEY_DOWN_SERVO 614
#define LEFT_KEY_UP_SERVO 245
#define RIGHT_KEY_DOWN_SERVO 614
#define RIGHT_KEY_UP_SERVO 220

// limit switch connections, any digital pins will do.
#define LEFT_LIMIT_SWITCH_PIN 8
#define RIGHT_LIMIT_SWITCH_PIN 9

// stepper motor connections, any digital pins will do.
#define LEFT_STEPPER_DIRECTION_PIN 2
#define LEFT_STEPPER_STEP_PIN 3
#define RIGHT_STEPPER_DIRECTION_PIN 4
#define RIGHT_STEPPER_STEP_PIN 5
#define LEFT_STEPPER_ENABLE 10
#define RIGHT_STEPPER_ENABLE 11

// PCA settings
#define nbPCAServo 16

// Connect both pca driver to same scl sda pins.
// Define PCA addressing
#define LEFT_PCA_ADDRESS 0x40
#define RIGHT_PCA_ADDRESS 0x41

// stepper motor settings and steps per revolution
#define motorInterfaceType 1
#define pulserate 8000
#define accelrate 8000

// Global Objects
Adafruit_PWMServoDriver pcal = Adafruit_PWMServoDriver(LEFT_PCA_ADDRESS);
Adafruit_PWMServoDriver pcar = Adafruit_PWMServoDriver(RIGHT_PCA_ADDRESS);

AccelStepper stepperl = AccelStepper(motorInterfaceType, LEFT_STEPPER_STEP_PIN, LEFT_STEPPER_DIRECTION_PIN);
AccelStepper stepperr = AccelStepper(motorInterfaceType, RIGHT_STEPPER_STEP_PIN, RIGHT_STEPPER_DIRECTION_PIN);

bool stepperl_on = true;
bool stepperr_on = true;


// extra note,
// the stepper motor for the left hand is actually placed on the right side.
// for positive counter, it moves the left hand to the left.
// the stepper motor for the right hand is actually placed on the left side.
// for positive counter, it moves the right hand to the right.

// song data are put into the following 3 integer arrays
// comment out example arrays once copy in the actual instructions
// eg:
// INSERT_INSTRUCTION_HERE
const PROGMEM int song_inst_count = 1994;
const PROGMEM long song_time[] ={ 0, 0, 2000, 2000, 2000, 2000, 2375, 2500, 2750, 2875, 3125, 3250, 3500, 3625, 3875, 4000, 4250, 4375, 4625, 4750, 5000, 5125, 5375, 5500, 5750, 5875, 6125, 6250, 6250, 6500, 6500, 6500, 6625, 6875, 7000, 7250, 7375, 7625, 7750, 8000, 8125, 8375, 8500, 8750, 8875, 9125, 9250, 9500, 9625, 9875, 10000, 10250, 10328, 10328, 10328, 10375, 10625, 10750, 11000, 11000, 11000, 11000, 11125, 11375, 11500, 11750, 11875, 12125, 12250, 12500, 12578, 12578, 12578, 12625, 12875, 13000, 13250, 13250, 13250, 13375, 13625, 13750, 14000, 14125, 14375, 14500, 14750, 14875, 15125, 15125, 15250, 15500, 15500, 15500, 15625, 15875, 16000, 16250, 16375, 16625, 16750, 17000, 17125, 17375, 17500, 17500, 17500, 17750, 17750, 17750, 17875, 18125, 18250, 18500, 18625, 18875, 19000, 19250, 19375, 19625, 19750, 19750, 19750, 20000, 20000, 20125, 20375, 20750, 20875, 21125, 21250, 21500, 21625, 21875, 22000, 22250, 22375, 22625, 22750, 23000, 23125, 23375, 23375, 23500, 23750, 23875, 23875, 24125, 24250, 24250, 24250, 24250, 24375, 24500, 24500, 24500, 24500, 24625, 24875, 25000, 25250, 25375, 25625, 25750, 26000, 26125, 26375, 26500, 26750, 26875, 27125, 27250, 27500, 27625, 27625, 27875, 27875, 28000, 28250, 28375, 28375, 28625, 28750, 28750, 28750, 28750, 28875, 29000, 29000, 29000, 29125, 29375, 29500, 29750, 29875, 30125, 30250, 30500, 30625, 30875, 31000, 31000, 31000, 31250, 31250, 31375, 31625, 31750, 32000, 32125, 32375, 32500, 32750, 32875, 33125, 33250, 33250, 33500, 33500, 33500, 33625, 33875, 34000, 34250, 34375, 34625, 34750, 35000, 35125, 35375, 35500, 35500, 35750, 35750, 35750, 35875, 36125, 36250, 36250, 36500, 36500, 36625, 36875, 37000, 37250, 37375, 37625, 37750, 37750, 38000, 38000, 38000, 38125, 38375, 38500, 38500, 38750, 38875, 39125, 39250, 39500, 39625, 39875, 40000, 40250, 40375, 40625, 40750, 41000, 41125, 41375, 41500, 41750, 41875, 42125, 42250, 42250, 42500, 42500, 42625, 42875, 43000, 43250, 43375, 43625, 43750, 44000, 44125, 44375, 44500, 44750, 44875, 45125, 45250, 45500, 45625, 45875, 45875, 46000, 46250, 46375, 46375, 46625, 46750, 46750, 46750, 46875, 47000, 47000, 47000, 47125, 47375, 47500, 47750, 47875, 48125, 48250, 48500, 48625, 48875, 49000, 49250, 49375, 49625, 49750, 50000, 50125, 50125, 50375, 50375, 50500, 50750, 50875, 50875, 51125, 51250, 51250, 51250, 51375, 51500, 51500, 51500, 51625, 51875, 52000, 52250, 52375, 52375, 52625, 52625, 52750, 53000, 53125, 53375, 53500, 53500, 53750, 53750, 53875, 54125, 54203, 54203, 54203, 54203, 54875, 54875, 54875, 55250, 55375, 55375, 55625, 55750, 55750, 56000, 56000, 56000, 56125, 56375, 56500, 56500, 56750, 56875, 57125, 57250, 57500, 57625, 57875, 58000, 58000, 58250, 58250, 58375, 58625, 58750, 59000, 59125, 59375, 59375, 59500, 59500, 59750, 59875, 60125, 60250, 60250, 60500, 60500, 60500, 60625, 60875, 61000, 61000, 61250, 61375, 61625, 61750, 62000, 62125, 62375, 62500, 62500, 62750, 62750, 62750, 62875, 63125, 63250, 63250, 63500, 63625, 63875, 64000, 64250, 64375, 64625, 64750, 64750, 65000, 65000, 65125, 65311, 65311, 66125, 66125, 66500, 66625, 66875, 67000, 67250, 67375, 67625, 67750, 68000, 68125, 68375, 68375, 68500, 68750, 68875, 69125, 69250, 69500, 69625, 69625, 69875, 70000, 70250, 70375, 70375, 70625, 70625, 70750, 71000, 71125, 71375, 71500, 71500, 71750, 71750, 71875, 72125, 72250, 72500, 72625, 72625, 72875, 72875, 72875, 73000, 73250, 73375, 73625, 73750, 73750, 73750, 74000, 74000, 74000, 74125, 74375, 74500, 74750, 74875, 75125, 75250, 75250, 75500, 75625, 75875, 76000, 76250, 76375, 76625, 76750, 77000, 77125, 77375, 77375, 77500, 77750, 77875, 78125, 78250, 78500, 78625, 78625, 78875, 79000, 79250, 79375, 79375, 79625, 79625, 79750, 80000, 80125, 80375, 80500, 80500, 80750, 80750, 80875, 81125, 81250, 81500, 81625, 81625, 81875, 81875, 81875, 82000, 82250, 82375, 82625, 82750, 82750, 82750, 83000, 83000, 83000, 83125, 83375, 83500, 83750, 83875, 84125, 84250, 84500, 84625, 84875, 85000, 85000, 85000, 85250, 85250, 85250, 85250, 85375, 85625, 85750, 86000, 86125, 86375, 86500, 86750, 86875, 87125, 87250, 87250, 87250, 87250, 87500, 87500, 87500, 87625, 87875, 88000, 88250, 88375, 88625, 88703, 88703, 88703, 88703, 89375, 89375, 89500, 89750, 89750, 89750, 89875, 90125, 90250, 90500, 90625, 90875, 90881, 90881, 90881, 90881, 91625, 91625, 91750, 92000, 92000, 92125, 92375, 92500, 92750, 92875, 93125, 93250, 93500, 93625, 93875, 94000, 94000, 94250, 94250, 94250, 94375, 94625, 94750, 95000, 95125, 95375, 95500, 95750, 95875, 96125, 96250, 96250, 96250, 96500, 96500, 96625, 96875, 97000, 97250, 97375, 97750, 98000, 98375, 98500, 98500, 98750, 98750, 98875, 99125, 99250, 99500, 99625, 99875, 100000, 100250, 100375, 100625, 100750, 100750, 101000, 101000, 101000, 101125, 101375, 101500, 101750, 101750, 102125, 102250, 102500, 102625, 102875, 103000, 103114, 103114, 104000, 104000, 104375, 104375, 104500, 104750, 104875, 104875, 105125, 105250, 105250, 105250, 105250, 105375, 105500, 105500, 105500, 105625, 105875, 106000, 106250, 106375, 106625, 106750, 107000, 107125, 107375, 107500, 107750, 107875, 107875, 108125, 108250, 108500, 108625, 108875, 108875, 109000, 109250, 109375, 109375, 109625, 109750, 109750, 109750, 109875, 110000, 110000, 110000, 110125, 110375, 110500, 110500, 110750, 110875, 111125, 111250, 111500, 111625, 111875, 112000, 112000, 112250, 112250, 112250, 112375, 112625, 112750, 112750, 113000, 113125, 113125, 113375, 113375, 113375, 113500, 113750, 113875, 113875, 114125, 114250, 114250, 114500, 114500, 114500, 114625, 114875, 115000, 115250, 115375, 115625, 115750, 116000, 116125, 116375, 116500, 116750, 116875, 117125, 117250, 117500, 117625, 117625, 117625, 117875, 117875, 117875, 118000, 118250, 118375, 118625, 118750, 118750, 118750, 119000, 119000, 119000, 119125, 119375, 119500, 119750, 119875, 120125, 120250, 120500, 120625, 120875, 121000, 121000, 121000, 121250, 121250, 121375, 121625, 121750, 122000, 122125, 122125, 122375, 122375, 122375, 122375, 122500, 122750, 122875, 122875, 123125, 123250, 123250, 123250, 123500, 123500, 123500, 123625, 123875, 124000, 124250, 124375, 124625, 124750, 125000, 125125, 125375, 125500, 125750, 125875, 126125, 126250, 126500, 126625, 126875, 127000, 127250, 127375, 127625, 127750, 127750, 127750, 128000, 128000, 128125, 128156, 129113, 129125, 129500, 129625, 129875, 130000, 130250, 130375, 130625, 130750, 131000, 131125, 131375, 131500, 131750, 131875, 132125, 132149, 132149, 132149, 132250, 132250, 132500, 132500, 133250, 133250, 133625, 133750, 134000, 134125, 134375, 134500, 134750, 134875, 135125, 135250, 135500, 135625, 135875, 136000, 136250, 136260, 136260, 136260, 136750, 136750, 137000, 137000, 137360, 137375, 137750, 137875, 138125, 138250, 138500, 138625, 138875, 139000, 139250, 139375, 139625, 139750, 140000, 140125, 140375, 140500, 140750, 140875, 141125, 141250, 141250, 141250, 141500, 141500, 141625, 141875, 142250, 142375, 142625, 142750, 143000, 143125, 143167, 143167, 144125, 144125, 144500, 144625, 144875, 145000, 145250, 145375, 145625, 145750, 145750, 145750, 146000, 146000, 146125, 146375, 146875, 147125, 147500, 147625, 147875, 148000, 148250, 148375, 148625, 148750, 149000, 149125, 149500, 149750, 150125, 150250, 150250, 150250, 150500, 150500, 150500, 150625, 150875, 151000, 151250, 151375, 151625, 151750, 152000, 152024, 152024, 152024, 153125, 153125, 153500, 153625, 153875, 154000, 154250, 154375, 154625, 154750, 154750, 154750, 155000, 155000, 155125, 155375, 155875, 156125, 156500, 156625, 156875, 157000, 157250, 157375, 157625, 157750, 158000, 158125, 158375, 158500, 158750, 158875, 159250, 159250, 159250, 159500, 159875, 160000, 160250, 160375, 160625, 160750, 161000, 161023, 161023, 161023, 162123, 162125, 162500, 162625, 162739, 162739, 163625, 163625, 164000, 164125, 164375, 164500, 164750, 164875, 165125, 165250, 165500, 165625, 165875, 166000, 166250, 166328, 166328, 166328, 167000, 167000, 167500, 167750, 168125, 168250, 168500, 168500, 168625, 168875, 169250, 169375, 169625, 169750, 170000, 170125, 170375, 170500, 170875, 171125, 171500, 171625, 172000, 172250, 172625, 172750, 172750, 172750, 173000, 173000, 173125, 173375, 173750, 173875, 174125, 174250, 174500, 174625, 174875, 175000, 175375, 175625, 176000, 176125, 176500, 176750, 177125, 177250, 177250, 177250, 177500, 177500, 177625, 177875, 178250, 178375, 178625, 178750, 179000, 179125, 179375, 179500, 179500, 179500, 179750, 179875, 180125, 180500, 180625, 181000, 181250, 181625, 181750, 181750, 182000, 182125, 182375, 182750, 182875, 183250, 183500, 183875, 184000, 184000, 184250, 184250, 184375, 184625, 185000, 185125, 185500, 185750, 186125, 186250, 186250, 186250, 186500, 186500, 186500, 186506, 186506, 186506, 187250, 187250, 187625, 187750, 188000, 188125, 188375, 188500, 188750, 188875, 189125, 189250, 189500, 189625, 189875, 189875, 190000, 190250, 190375, 190375, 190625, 190750, 190750, 190750, 190750, 190875, 191000, 191000, 191000, 191000, 191125, 191375, 191500, 191750, 191875, 192125, 192250, 192500, 192625, 192875, 193000, 193250, 193375, 193625, 193750, 194000, 194125, 194125, 194375, 194375, 194500, 194750, 194875, 194875, 195125, 195250, 195250, 195250, 195250, 195375, 195500, 195500, 195500, 195625, 195875, 196000, 196250, 196375, 196625, 196750, 197000, 197125, 197375, 197500, 197500, 197500, 197750, 197750, 197875, 198125, 198250, 198500, 198625, 198875, 199000, 199250, 199375, 199625, 199750, 199750, 200000, 200000, 200000, 200125, 200375, 200500, 200750, 200875, 201125, 201250, 201500, 201625, 201875, 202000, 202000, 202250, 202250, 202250, 202375, 202625, 202750, 202750, 203000, 203125, 203125, 203125, 203375, 203375, 203500, 203750, 203875, 204125, 204250, 204250, 204500, 204500, 204500, 204625, 204875, 205000, 205000, 205250, 205375, 205625, 205750, 206000, 206125, 206375, 206500, 206750, 206875, 207125, 207130, 207130, 207130, 207873, 207875, 207875, 208250, 208375, 208375, 208625, 208750, 208750, 208750, 208875, 209000, 209000, 209000, 209125, 209375, 209500, 209750, 209875, 210125, 210250, 210500, 210625, 210875, 211000, 211250, 211375, 211625, 211750, 212000, 212125, 212125, 212375, 212375, 212500, 212750, 212875, 212875, 213125, 213250, 213250, 213250, 213375, 213500, 213500, 213500, 213625, 213875, 214000, 214250, 214375, 214625, 214703, 214703, 214703, 214703, 215375, 215375, 215500, 215750, 215750, 215750, 215875, 216125, 216250, 216500, 216625, 216625, 216625, 216875, 216875, 216875, 217000, 217250, 217375, 217625, 217750, 217750, 217750, 218000, 218000, 218000, 218125, 218375, 218500, 218750, 218875, 219125, 219250, 219500, 219625, 219875, 220000, 220000, 220000, 220250, 220250, 220250, 220375, 220625, 220750, 221000, 221125, 221375, 221500, 221750, 221875, 222125, 222250, 222250, 222250, 222500, 222500, 222625, 222875, 223000, 223250, 223375, 223625, 223750, 224000, 224125, 224375, 224500, 224500, 224750, 224750, 224750, 224750, 224875, 225125, 225250, 225500, 225625, 225875, 226000, 226250, 226375, 226625, 226750, 226750, 226750, 226750, 227000, 227000, 227000, 227125, 227375, 227500, 227750, 227875, 228125, 228250, 228500, 228625, 228875, 229000, 229250, 229375, 229625, 229750, 230000, 230125, 230125, 230375, 230375, 230500, 230750, 230875, 230875, 231125, 231250, 231500, 231625, 231875, 232000, 232250, 232375, 232375, 232625, 232625, 232750, 233000, 233125, 233375, 233500, 233500, 233750, 233875, 234125, 234250, 234500, 234625, 234875, 234875, 234875, 235000, 235250, 235375, 235625, 235750, 235750, 235750, 236000, 236000, 236000, 236125, 236375, 236500, 236750, 236875, 237125, 237250, 237500, 237625, 237875, 238000, 238000, 238250, 238250, 238375, 238625, 238750, 239000, 239125, 239125, 239375, 239500, 239750, 239875, 240125, 240250, 240250, 240500, 240500, 240625, 240875, 241000, 241250, 241375, 241375, 241625, 241750, 242000, 242125, 242375, 242500, 242750, 242750, 242750, 242875, 243125, 243250, 243250, 243500, 243625, 243625, 243875, 243875, 243875, 244000, 244250, 244375, 244625, 244750, 245000, 245125, 245375, 245500, 245750, 245875, 245875, 245875, 246125, 246125, 246125, 246250, 246500, 246625, 246875, 247000, 247250, 247375, 247625, 247703, 247703, 247703, 247703, 248125, 248375, 248375, 248375, 248375, 248750, 248875, 249125, 249250, 249500, 249625, 249625, 249875, 250000, 250250, 250375, 250375, 250625, 250625, 250625, 250750, 251000, 251078, 251078, 251078, 251078, 251500, 251750, 251750, 251750, 251750, 252125, 252250, 252250, 252500, 252625, 252625, 252875, 252875, 252875, 253000, 253250, 253328, 253328, 253328, 253328, 253750, 254000, 254000, 254000, 254000, 254375, 254500, 254750, 254875, 254875, 254875, 255125, 255125, 255125, 255203, 255203, 255203, 255203, 255875, 255875, 256000, 256250, 256250, 256250, 256375, 256625, 256750, 257000, 257125, 257375, 257453, 257453, 257453, 257453, 258125, 258125, 258250, 258500, 258500, 258500, 258500, 258625, 258875, 259000, 259250, 259375, 259375, 259375, 259375, 259625, 259625, 259750, 260000, 260125, 260375, 260500, 260500, 260750, 260750, 260750, 260750, 260875, 261125, 261250, 261500, 261625, 261875, 262000, 262250, 262375, 262625, 262750, 262750, 262750, 262750, 263000, 263000, 263000, 263125, 263375, 263750, 263875, 264250, 264500, 264875, 265000, 265000, 265000, 265000, 265250, 265250, 265250, 265375, 265625, 265750, 266000, 266125, 266375, 266500, 266750, 266875, 267125, 267250, 267500, 267625, 267875, 268000, 268250, 268375, 268375, 268375, 268625, 268625, 268750, 269000, 269125, 269125, 269375, 269500, 269500, 269625, 269750, 269750, 269750, 269875, 270125, 270250, 270500, 270625, 270875, 271000, 271250, 271375, 271625, 271750, 272000, 272125, 272375, 272500, 272750, 272875, 272875, 272875, 273125, 273125, 273250, 273500, 273625, 273750, 273875, 274000, 274000, 274036, 274036, 274125, 274250, 274250, 274993, 275000, 275375, 275500, 275750, 275875, 276125, 276250, 276500, 276625, 276875, 277000, 277000, 277000, 277250, 277375, 277625, 277703, 277703, 277750, 278125, 278250, 278375, 278375, 278375, 278375, 278750, 278875, 279125, 279250, 279500, 279625, 279875, 280000, 280250, 280346, 280346, 280346, 281375, 281375, 281750, 281875, 282125, 282250, 282375, 282375, 282500, 282625, 282625, 282750, 282875, 282875, 282875, 283000, 283250, 283375, 283625, 283750, 283787, 283787, 284002, 284744, 284746, 284750, 285125, 285250, 285500, 285625, 285875, 286000, 286000, 286000, 286250, 286375, 286625, 286703, 286703, 286750, 287125, 287250, 287375, 287375, 287375, 287375, 287750, 287875, 288125, 288250, 288500, 288625, 288875, 289000, 289250, 289343, 289343, 289343, 290372, 290375, 290750, 290875, 291125, 291250, 291375, 291375, 291500, 291625, 291625, 291750, 291875, 291875, 291875, 291875, 292000, 292250, 292375, 292375, 292625, 292750, 293000, 293125, 293375, 293500, 293750, 293875, 294250, 294500, 294875, 295000, 295250, 295375, 295625, 295750, 296000, 296125, 296125, 296375, 296500, 296875, 297125, 297500, 297625, 297875, 298000, 298375, 298625, 299125, 299375, 299750, 299875, 300125, 300250, 300500, 300625, 300625, 300875, 301000, 302000, 302250, 302250, 302250, 302250, 302250, 304750, 304750, 304750, 304750, 304750, 305000, 305000, 305000, 305000, 305000, 309125, 309125, 309125, 309125, 309125 };
const PROGMEM int song_sync[] ={ 0, 1, 7, 6, 2, 3, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 4, 4, 0, 5, 7, 5, 7, 6, 6, 2, 5, 7, 5, 7, 5, 7, 5, 7, 4, 4, 0, 5, 7, 5, 7, 6, 2, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 5, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 6, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 4, 7, 5, 7, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 4, 7, 5, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 7, 5, 7, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 7, 5, 7, 6, 7, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 5, 5, 1, 7, 7, 2, 7, 5, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 7, 6, 5, 4, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 5, 5, 1, 7, 2, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 7, 5, 4, 5, 7, 7, 6, 5, 7, 5, 7, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 5, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 5, 5, 1, 7, 2, 4, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 5, 5, 1, 7, 2, 4, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 5, 7, 5, 7, 5, 5, 7, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 6, 5, 7, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 1, 7, 2, 7, 7, 5, 7, 5, 5, 7, 5, 4, 4, 7, 5, 7, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 7, 5, 7, 7, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 6, 6, 5, 7, 5, 7, 5, 4, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 7, 6, 6, 5, 7, 5, 5, 7, 5, 4, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 6, 5, 1, 2, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 4, 4, 6, 6, 7, 2, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 4, 4, 6, 6, 2, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 6, 5, 7, 7, 5, 7, 5, 7, 5, 5, 1, 7, 2, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 6, 5, 7, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 4, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 7, 2, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 6, 5, 7, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 4, 4, 7, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 2, 7, 7, 5, 5, 1, 7, 2, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 7, 2, 5, 7, 7, 5, 6, 6, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 4, 4, 6, 6, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 5, 7, 7, 5, 4, 4, 6, 6, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 5, 7, 7, 5, 5, 7, 7, 5, 4, 6, 5, 7, 7, 5, 5, 7, 7, 5, 4, 6, 6, 5, 7, 7, 5, 5, 7, 7, 5, 4, 4, 7, 6, 6, 5, 5, 1, 7, 2, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 4, 7, 5, 7, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 4, 7, 5, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 4, 5, 7, 6, 5, 7, 5, 7, 5, 4, 7, 7, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 2, 7, 7, 7, 5, 5, 7, 5, 4, 7, 5, 7, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 5, 7, 5, 4, 7, 5, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 5, 5, 1, 7, 2, 4, 7, 6, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 5, 7, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 5, 7, 5, 7, 5, 7, 6, 7, 5, 7, 5, 7, 5, 4, 5, 7, 7, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 4, 7, 6, 5, 7, 5, 7, 5, 4, 7, 5, 7, 5, 7, 5, 7, 7, 6, 5, 7, 5, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 5, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 5, 1, 4, 7, 6, 7, 2, 7, 5, 7, 5, 7, 5, 4, 7, 5, 7, 5, 5, 7, 6, 7, 5, 7, 5, 5, 5, 1, 4, 7, 7, 6, 2, 7, 5, 5, 7, 5, 4, 7, 6, 7, 5, 7, 5, 5, 5, 1, 4, 7, 6, 7, 2, 7, 5, 7, 5, 4, 5, 7, 6, 7, 5, 5, 5, 1, 7, 2, 4, 7, 6, 7, 5, 7, 5, 7, 5, 7, 5, 5, 5, 1, 7, 2, 4, 7, 6, 6, 7, 5, 7, 5, 7, 5, 4, 4, 5, 7, 6, 5, 7, 5, 7, 5, 4, 7, 6, 6, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 5, 6, 6, 7, 5, 7, 7, 5, 5, 7, 7, 5, 4, 4, 5, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 7, 5, 7, 4, 5, 7, 5, 6, 4, 7, 6, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 7, 6, 5, 7, 5, 4, 7, 5, 6, 5, 1, 4, 6, 6, 2, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 4, 4, 6, 5, 7, 5, 1, 4, 6, 4, 7, 6, 6, 2, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 7, 2, 7, 5, 7, 5, 4, 4, 7, 5, 6, 4, 7, 6, 6, 5, 7, 5, 7, 5, 5, 1, 1, 2, 2, 7, 7, 5, 7, 5, 7, 5, 4, 4, 6, 5, 7, 5, 1, 4, 6, 4, 7, 6, 6, 2, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 1, 2, 7, 7, 5, 7, 5, 4, 4, 7, 5, 6, 4, 7, 7, 6, 6, 5, 7, 5, 5, 7, 5, 7, 5, 7, 5, 7, 5, 5, 7, 7, 5, 7, 5, 7, 5, 7, 5, 4, 6, 5, 4, 7, 7, 5, 7, 5, 5, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 4, 4, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4, 7, 7, 7, 6, 6, 5, 5, 5, 4, 4 };
const PROGMEM int song_args[] ={ 299, 748, 1, 3, 0, 0, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 3, 1, 0, 14, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 0, 14, 262, 1, 10, 7, 2, 0, 14, 0, 10, 7, 2, 10, 7, 2, 10, 7, 0, 14, 224, 2, 10, 7, 2, 13, 0, 10, 8, 2, 13, 8, 2, 13, 8, 2, 13, 13, 8, 1, 1, 15, 13, 6, 1, 13, 6, 1, 13, 7, 1, 10, 7, 1, 15, 1, 15, 1, 10, 7, 1, 9, 7, 1, 9, 6, 1, 9, 6, 15, 1, 7, 15, 9, 1, 7, 1, 1, 7, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 15, 1, 10, 7, 15, 1, 10, 7, 7, 15, 15, 15, 1, 15, 6, 15, 10, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 15, 1, 15, 13, 9, 1, 15, 13, 9, 6, 15, 15, 15, 1, 7, 15, 13, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 7, 15, 2, 13, 10, 7, 2, 13, 7, 2, 13, 7, 2, 13, 7, 13, 1, 4, 15, 13, 4, 1, 10, 4, 1, 10, 4, 1, 10, 4, 4, 2, 13, 4, 10, 4, 2, 13, 15, 9, 4, 2, 9, 4, 2, 9, 4, 4, 1, 10, 10, 9, 4, 1, 10, 10, 4, 1, 10, 4, 1, 10, 4, 1, 10, 4, 1, 10, 4, 1, 10, 4, 1, 10, 4, 10, 0, 10, 10, 4, 0, 10, 4, 0, 10, 4, 0, 10, 4, 0, 10, 4, 0, 10, 4, 14, 0, 10, 4, 14, 0, 10, 4, 10, 14, 14, 0, 14, 8, 10, 4, 0, 12, 4, 0, 12, 4, 0, 12, 4, 0, 12, 4, 0, 12, 4, 14, 14, 0, 12, 4, 14, 0, 12, 4, 8, 14, 14, 0, 6, 14, 12, 6, 0, 10, 6, 6, 0, 4, 10, 4, 0, 10, 4, 4, 0, 3, 10, 7, 0, 7, 14, 711, 1, 15, 0, 9, 1, 15, 12, 9, 3, 1, 15, 4, 12, 6, 1, 15, 10, 6, 1, 10, 6, 1, 10, 6, 4, 2, 10, 10, 6, 2, 9, 6, 12, 14, 9, 10, 6, 12, 9, 6, 14, 1, 15, 13, 9, 6, 1, 15, 10, 6, 1, 10, 6, 1, 10, 6, 13, 1, 15, 13, 10, 5, 1, 15, 9, 5, 1, 9, 5, 1, 9, 5, 13, 6, 4, 9, 6, 823, 0, 0, 4, 0, 9, 4, 0, 9, 5, 0, 9, 5, 0, 14, 9, 5, 0, 9, 5, 0, 9, 14, 6, 0, 10, 6, 4, 0, 10, 10, 6, 0, 10, 6, 10, 0, 14, 10, 6, 0, 10, 6, 14, 0, 10, 13, 10, 6, 0, 10, 6, 10, 13, 0, 14, 4, 10, 5, 0, 9, 5, 0, 9, 14, 5, 0, 9, 5, 0, 9, 5, 0, 9, 5, 0, 14, 9, 5, 0, 9, 5, 0, 9, 14, 6, 0, 10, 6, 4, 0, 10, 10, 6, 0, 10, 6, 10, 0, 14, 10, 6, 0, 10, 6, 14, 0, 10, 13, 10, 6, 0, 10, 6, 10, 13, 0, 4, 14, 10, 5, 0, 9, 5, 0, 9, 5, 0, 9, 5, 4, 14, 0, 1, 15, 14, 9, 4, 0, 8, 4, 0, 8, 4, 0, 8, 4, 1, 15, 14, 0, 12, 14, 8, 3, 0, 11, 3, 0, 11, 0, 14, 785, 13, 0, 12, 0, 13, 14, 13, 5, 0, 11, 5, 0, 11, 0, 14, 711, 15, 0, 13, 2, 4, 15, 6, 2, 10, 6, 2, 10, 6, 2, 10, 6, 4, 1, 6, 15, 10, 4, 1, 11, 4, 1, 11, 4, 1, 11, 4, 6, 15, 9, 7, 11, 1, 9, 4, 1, 4, 1, 4, 1, 7, 9, 7, 4, 1, 9, 3, 1, 9, 3, 0, 9, 3, 0, 7, 1, 7, 13, 3, 4, 1, 4, 9, 4, 9, 9, 4, 15, 9, 15, 860, 10, 0, 1, 15, 10, 7, 1, 15, 10, 7, 7, 13, 15, 15, 1, 15, 7, 10, 9, 1, 12, 9, 1, 12, 9, 1, 12, 9, 1, 12, 15, 9, 1, 12, 9, 1, 15, 12, 9, 1, 15, 12, 9, 7, 15, 15, 1, 15, 13, 12, 7, 1, 15, 10, 7, 1, 10, 7, 1, 10, 7, 13, 0, 14, 9, 10, 7, 0, 14, 10, 7, 9, 1, 15, 7, 10, 7, 1, 15, 10, 7, 7, 3, 6, 15, 10, 7, 3, 9, 7, 3, 9, 7, 3, 9, 7, 3, 9, 7, 3, 9, 7, 6, 15, 3, 6, 15, 9, 7, 3, 9, 7, 6, 15, 4, 7, 15, 9, 9, 4, 15, 9, 4, 15, 9, 4, 15, 9, 7, 15, 3, 13, 15, 7, 3, 10, 7, 13, 1, 15, 0, 14, 10, 4, 1, 15, 11, 4, 0, 14, 14, 15, 1, 11, 0, 14, 3, 0, 9, 3, 0, 9, 3, 0, 10, 3, 0, 10, 3, 0, 7, 3, 0, 7, 3, 0, 15, 1, 15, 1, 3, 673, 0, 5, 10, 5, 13, 10, 6, 13, 10, 6, 13, 10, 3, 13, 10, 3, 13, 10, 13, 935, 15, 1, 15, 1, 5, 0, 11, 5, 0, 11, 5, 0, 14, 5, 0, 14, 5, 0, 11, 5, 0, 11, 0, 673, 15, 1, 15, 1, 0, 0, 5, 0, 11, 5, 0, 11, 5, 0, 14, 5, 0, 14, 5, 0, 11, 5, 0, 11, 5, 0, 15, 1, 1, 15, 5, 6, 3, 6, 10, 3, 6, 10, 6, 860, 7, 0, 3, 7, 10, 3, 7, 10, 14, 7, 1, 15, 15, 1, 14, 1, 1, 4, 1, 4, 9, 1, 4, 9, 15, 4, 9, 15, 9, 15, 9, 15, 15, 1, 1, 15, 1, 9, 8, 1, 4, 8, 11, 4, 8, 11, 8, 1122, 4, 0, 1, 4, 8, 1, 4, 8, 11, 4, 15, 1, 1, 15, 11, 0, 0, 3, 0, 3, 7, 0, 3, 7, 10, 3, 7, 10, 14, 7, 10, 14, 10, 1, 15, 14, 7, 14, 10, 7, 3, 10, 7, 3, 7, 860, 0, 10, 14, 10, 14, 711, 11, 0, 15, 11, 8, 15, 11, 8, 4, 11, 8, 4, 1, 8, 4, 1, 4, 673, 3, 0, 3, 3, 6, 3, 1, 14, 6, 3, 5, 3, 6, 5, 5, 6, 3, 5, 3, 3, 6, 3, 6, 3, 6, 3, 1, 14, 15, 1, 6, 3, 5, 3, 6, 5, 5, 6, 3, 5, 3, 3, 6, 3, 6, 3, 6, 3, 15, 1, 15, 1, 6, 3, 5, 3, 6, 5, 5, 6, 3, 5, 15, 1, 2, 3, 0, 11, 0, 11, 0, 11, 0, 2, 13, 11, 6, 11, 6, 11, 6, 11, 6, 13, 15, 1, 11, 5, 10, 5, 10, 3, 10, 3, 15, 1, 0, 7, 15, 10, 0, 748, 7, 0, 1, 7, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 15, 10, 7, 1, 15, 10, 7, 7, 15, 15, 15, 1, 15, 6, 15, 10, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 15, 1, 15, 13, 9, 1, 15, 13, 9, 6, 15, 15, 15, 1, 7, 15, 13, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 7, 15, 2, 13, 10, 7, 2, 13, 7, 2, 13, 7, 2, 13, 7, 13, 1, 4, 15, 13, 4, 1, 10, 4, 1, 10, 4, 1, 10, 4, 4, 2, 13, 4, 10, 4, 2, 13, 9, 4, 4, 15, 2, 4, 9, 4, 2, 9, 4, 4, 1, 10, 10, 9, 4, 1, 10, 10, 4, 4, 10, 10, 4, 15, 10, 4, 15, 10, 4, 10, 823, 0, 0, 14, 6, 0, 14, 11, 6, 10, 14, 14, 0, 14, 9, 11, 9, 0, 12, 9, 0, 12, 9, 0, 12, 9, 0, 12, 9, 0, 12, 9, 14, 0, 14, 12, 9, 0, 14, 12, 9, 9, 14, 14, 0, 10, 14, 12, 6, 0, 11, 6, 0, 11, 0, 14, 860, 9, 0, 10, 0, 9, 14, 9, 7, 0, 9, 7, 9, 14, 1, 7, 15, 9, 4, 1, 9, 4, 7, 15, 3, 6, 15, 9, 7, 3, 9, 7, 3, 9, 7, 3, 9, 7, 6, 15, 4, 7, 15, 9, 9, 4, 15, 9, 4, 15, 9, 4, 15, 9, 7, 15, 2, 13, 15, 7, 2, 10, 7, 2, 10, 7, 2, 10, 7, 13, 0, 1, 15, 14, 10, 7, 0, 9, 7, 0, 9, 7, 0, 9, 7, 1, 15, 14, 1, 15, 7, 9, 4, 1, 9, 4, 1, 9, 4, 1, 9, 4, 1, 9, 6, 1, 9, 6, 15, 1, 15, 9, 6, 1, 15, 9, 6, 1, 9, 7, 1, 10, 7, 7, 1, 13, 10, 7, 1, 10, 7, 13, 1, 10, 7, 1, 10, 7, 1, 13, 14, 10, 7, 1, 10, 7, 13, 14, 1, 15, 7, 10, 6, 1, 9, 6, 1, 9, 6, 1, 9, 6, 15, 1, 15, 9, 6, 1, 9, 6, 15, 1, 9, 7, 1, 10, 7, 7, 1, 13, 10, 7, 1, 10, 7, 13, 1, 10, 7, 1, 10, 7, 1, 14, 13, 10, 7, 1, 14, 10, 7, 13, 1, 7, 15, 10, 6, 1, 9, 6, 1, 9, 6, 1, 9, 6, 7, 15, 1, 13, 15, 9, 7, 1, 10, 7, 1, 10, 7, 1, 7, 15, 823, 13, 0, 9, 14, 0, 9, 0, 12, 9, 0, 12, 9, 9, 0, 12, 9, 14, 0, 10, 14, 12, 6, 0, 6, 14, 785, 10, 0, 14, 7, 0, 8, 0, 14, 13, 8, 7, 0, 9, 14, 13, 7, 0, 7, 14, 748, 9, 1, 6, 15, 0, 9, 1, 13, 9, 6, 15, 1, 7, 15, 13, 1, 15, 711, 12, 0, 7, 1, 2, 15, 12, 9, 1, 11, 9, 1, 11, 1, 15, 748, 9, 0, 2, 1, 1, 15, 15, 9, 7, 1, 9, 7, 1, 15, 15, 2, 13, 9, 7, 2, 9, 7, 13, 1, 1, 15, 15, 9, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 1, 15, 15, 1, 15, 15, 10, 6, 9, 6, 9, 6, 9, 6, 1, 15, 15, 7, 7, 15, 9, 1, 7, 7, 1, 1, 7, 7, 1, 10, 7, 1, 10, 7, 1, 10, 7, 7, 15, 15, 1, 10, 7, 15, 1, 10, 7, 15, 15, 1, 15, 6, 10, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 1, 13, 9, 15, 6, 1, 15, 13, 9, 1, 15, 13, 9, 15, 13, 935, 15, 7, 15, 0, 5, 0, 5, 11, 0, 5, 11, 14, 5, 11, 14, 7, 15, 15, 11, 14, 14, 898, 15, 15, 15, 12, 1, 15, 0, 15, 12, 8, 15, 12, 8, 5, 12, 8, 5, 8, 673, 6, 0, 10, 6, 5, 10, 1, 15, 3, 5, 15, 15, 0, 7, 15, 3, 14, 0, 11, 14, 11, 860, 935, 0, 0, 11, 5, 11, 14, 5, 11, 14, 7, 15, 15, 11, 14, 14, 898, 15, 15, 15, 12, 1, 15, 0, 15, 12, 8, 15, 12, 8, 5, 12, 8, 5, 8, 673, 0, 6, 10, 6, 5, 10, 1, 15, 3, 5, 15, 15, 11, 0, 15, 7, 3, 5, 11, 0, 11, 5, 14, 11, 11, 14, 5, 11, 5, 0, 5, 0, 11, 5, 5, 11, 0, 5, 15, 15, 0, 15, 0, 5, 0, 0, 5, 0, 15, 15, 15, 10, 15, 15, 10, 10, 15, 7, 7, 10, 7, 11, 5, 0, 15, 7, 11, 5, 0, 15, 7, 11, 5, 0, 15, 7, 11, 5, 0, 15, 7 };



void setup()
{

    // Init Serial USB
    Serial.begin(9600);
    Serial.println(F("Initialize System"));
    // Init stepper driver
    pinMode(LEFT_STEPPER_ENABLE,OUTPUT);
    digitalWrite(LEFT_STEPPER_ENABLE,HIGH);
    stepperl.setMaxSpeed(pulserate);
    stepperl.setAcceleration(accelrate);

    
    pinMode(RIGHT_STEPPER_ENABLE,OUTPUT);
    digitalWrite(RIGHT_STEPPER_ENABLE,HIGH);
    stepperr.setMaxSpeed(pulserate);
    stepperr.setAcceleration(accelrate);
    
    
    // Init PCA/servo driver
    pcal.begin();
    pcal.setPWMFreq(60);
    pcar.begin();
    pcar.setPWMFreq(60);

    // Init limit switch.
    pinMode(LEFT_LIMIT_SWITCH_PIN, INPUT_PULLUP);
    pinMode(RIGHT_LIMIT_SWITCH_PIN, INPUT_PULLUP);

    // Init on board LED
    pinMode(LED_BUILTIN, OUTPUT);

    // indicate it
    Serial.print("Initialization Complete\n");
}

void loop()
{
    Serial.print("Loop Begin\n");
    resetServo();
    resetStepper();
    playSong();
}

void resetServo()
{
    // Lift All Fingers
    Serial.print("Reseting Servo Motors on left hand\n");
    for (int i = 0; i < nbPCAServo; i++)
    {
        pcal.setPWM(i, 0, LEFT_KEY_DOWN_SERVO);
        delay(300);
        pcal.setPWM(i, 0, LEFT_KEY_UP_SERVO);
    }
    delay(500);
    // Turn them all to not powered state. (potential auto reset, need to check)
    for (int i = 0; i < nbPCAServo; i++)
    {
        pcal.setPWM(i, 0, 0);
    }
    // repeat for the other hand
    Serial.print("Reseting Servo Motors on right hand\n");
    for (int i = 0; i < nbPCAServo; i++)
    {
        pcar.setPWM(i, 0, RIGHT_KEY_DOWN_SERVO);
        delay(300);
        pcar.setPWM(i, 0, RIGHT_KEY_UP_SERVO);
    }
    delay(500);
    for (int i = 0; i < nbPCAServo; i++)
    {
        pcar.setPWM(i, 0, 0);
    }
}

void resetStepper()
{
    int counter = 0;
    // reset main stepper motor
    Serial.print("Reseting Left Stepper Motors\n");
    stepperl.enableOutputs();
    stepperl_on=true;
    stepperl.setSpeed(-pulserate);
    stepperl.moveTo(4000); // this largest number of steps before hitting the end. calculated from 2 meter and 100 step per 6 cm.
    while (1)
    {
        stepperl.run(); // non-blocking operation, no delay required.
        if (digitalRead(LEFT_LIMIT_SWITCH_PIN) == LOW)
        {
            counter += 1;
        }
        else
        {
            counter = 0;
        }
        Serial.print(counter);
        if (counter >= 5)
        {
            break;
        }
    }
    Serial.print("\n");
    stepperl.setCurrentPosition(LEFT_LIMIT_SWTICH_RELATIVE_TO_A0);

    Serial.print("Reseting Right Stepper Motors\n");
    counter = 0;
    stepperr.enableOutputs();
    stepperr_on=true;
    stepperr.setSpeed(-pulserate);
    stepperr.moveTo(4000); // this largest number of steps before hitting the end. calculated from 2 meter and 100 step per 6 cm.
    while (1)
    {
        stepperr.run(); // non-blocking operation, no delay required.
        if (digitalRead(RIGHT_LIMIT_SWITCH_PIN) == LOW)
        {
            counter += 1;
        }
        else
        {
            counter = 0;
        }
        Serial.print(counter);
        if (counter >= 5)
        {
            break;
        }
    }
    Serial.print("\n");
    stepperr.setCurrentPosition(RIGHT_LIMIT_SWTICH_RELATIVE_TO_A0);
}

void playSong()
{
    Serial.print("Starting pLaying the song.\n");
    int instruction_index = 0;
    int tick = millis();
    while (instruction_index < song_inst_count)
    {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        if (millis() - tick >= pgm_read_dword_near(song_time + instruction_index))
        {
            int support_arg = pgm_read_word_near(song_args + instruction_index);
            switch (pgm_read_word_near(song_sync + instruction_index))
            {
            case 0:
                /* move left hand to x*/
                Serial.print("left to");
                stepperl_on = true;
                stepperl.enableOutputs();
                digitalWrite(LEFT_STEPPER_ENABLE,HIGH);
                stepperl.moveTo(-(support_arg - 87));
                break;
            case 1:
                /* move right hand to x*/
                Serial.print("right to ");
                stepperr_on = true;
                stepperr.enableOutputs();
                digitalWrite(RIGHT_STEPPER_ENABLE,HIGH);
                stepperr.moveTo((support_arg - 87));
                break;
            case 2:
                /* move left hand to x*/
                Serial.print("left stop");
                stepperr_on = false;
                stepperl.disableOutputs();
                digitalWrite(LEFT_STEPPER_ENABLE,LOW);
                break;
            case 3:
                /* move right hand to x*/
                Serial.print("right stop");
                stepperl_on = false;
                stepperr.disableOutputs();
                digitalWrite(RIGHT_STEPPER_ENABLE,LOW);
                break;
            case 4:
                /* release x key of left hand */
                pcal.setPWM(support_arg, 0, LEFT_KEY_UP_SERVO);
                break;
            case 5:
                /* release x key of right hand */
                pcar.setPWM(support_arg, 0, RIGHT_KEY_UP_SERVO);
                break;
            case 6:
                /* press x key of left hand */
                pcal.setPWM(support_arg, 0, LEFT_KEY_DOWN_SERVO);
                break;
            case 7:
                /* press x key of right hand */
                pcar.setPWM(support_arg, 0, RIGHT_KEY_DOWN_SERVO);
                break;
            }
            instruction_index++;
        }
        if (stepperl_on) {
          stepperl.run();
        }
        if (stepperr_on) {
          stepperr.run();
        }
        delayMicroseconds(500);
    }
}
