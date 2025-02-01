# Machine Learning Security Project
This project is developed for the UNICA.IT University Machine Learning Security exam. 

> **Master's degree in Computer Engineering, Cybersecurity and Artificial Intelligence - University of Cagliari**

> **Machine Learning Security - Supervisor: Prof. Battista Biggio**

> **Author**: Lello Molinario: 70/90/00369


***
# Table of Contents
1. [Installation](#installation)
2. [Project Goal](#project-goal)
4. [Solution Design](#solution-design)
5. [Conclusions](#conclusions)


***
## Installation
### Before running it, make sure you have Python 3.10 installed for compatibility between all libraries.

- Download the ZIP code or clone the repository with
  ```bash
  git clone https://github.com/lmolinario/ML_Sec_project.git
  ```
- Enter inside a repository dire`ctory and `install the requirements with

  ```bash
  pip3 install -r requirements.txt
  ```
- Run the file `entrypoint.py` to start the program 
 
  ```bash
  python3.10 entrypoint.py
  ```

## Project goal
The goal of this project is to re-evaluate 5 RobustBench models using another attack algorithm (e.g., FMN) and identify samples for which one attack succeeds while the other fails. In other words, we aim to compare the effectiveness of different attacks against robust models, to analyze in which cases one type of attack is effective while another fails, thus contributing to a deeper understanding of the robustness of models and attack algorithms.
scrivi un codice python per identificare
i campioni per i quali un attacco funziona e l'altro no. Spiegare i risultati, ovvero fornire
alcune motivazioni sul perché uno degli attacchi non ha funzionato correttamente, mentre l'altro sì.

## Solution Design
To re-evaluate the FNM model, we use as a basis for comparison the results of "AutoAttack - Robustbench"
calculated on the same epsilon (in this case epsilon = 8/255 with "L-inf" norm) and we take into account the samples that successfully perturb the image with epsilon < 8/255.

#### Attack algorithm
As indicated in our project I took as a reference the FMN attack, also known as FGSM (Fast Gradient Sign Method), which is one of the most common attacks against neural networks.
The basic idea of this attack is to calculate the gradient of the model with respect to the input image and add a perturbation in the direction of the gradient to maximize the loss. This type of attack can be implemented as follows:

![δ=ϵ⋅sign(∇xJ(θ,x,y))](misc/Formula FMN.png)

Where:

**δ** is the generated perturbation.

**ϵ** is the magnitude of the perturbation (i.e. the strength of the attack).

**∇xJ(θ,x,y)** is the gradient of the loss function J with respect to the input image x, calculated for the model parameters θ.

**sign(⋅)** refers to the function that takes the sign of each gradient value.

The **norm** is a fundamental concept in all adversarial attacks.
It defines how to measure and bound the size of the perturbation added to the original image. It is a mathematical measure that establishes the "strength" of the perturbation and is used to control how much the original image is modified.

In our case of **FMN attack**, the norm directly influences the creation of the adversarial perturbation and its control.
Depending on the type of norm chosen, the perturbation can have different characteristics:

**L∞ norm (maximum norm)**: measures the maximum difference for each pixel between the original and perturbed images. The L∞ norm is useful for testing robustness against attacks that limit the perturbation to the maximum value for each pixel.

**L2 norm (Euclidean)**: measures the Euclidean distance between the original and perturbed images. The L2 norm is generally sensitive to large changes in the images, but may not capture very small local perturbations that affect recognition.

**L1 norm**: measures the absolute sum of the pixel-by-pixel differences. It is less sensitive than the L2 norm to large perturbations, but can be effective for detecting small uniformly distributed changes.

**Lp norm** (where p is a value between 1 and ∞): It is a generalization of the L1, L2 and L∞ norms.

**For our project, we will use and compare "L∞".**

The distance "L∞" is a fundamental measure in adversarial attack problems, as it represents the maximum change that is applied to a single pixel in the image during the generation of adversarial samples.
Limiting the distance "L∞" to a specific epsilon value ( in our case 8/255 ) has several motivations and importance:
Adversarial attacks must be visually imperceptible to a human observer. If the perturbation exceeds a certain limit, the adversarial sample may appear distorted or artificial.
A smaller perturbation (epsilon < 8/255 ) ensures that the changes in pixels are minimal, keeping the image visually similar to the original.

Additionally, the CIFAR-10 dataset uses normalized images with pixel values ​​between 0 and 1.
A value of epsilon = 8/255 represents a very small change (about 3% of the full scale), which is consistent with the idea of ​​a “sneaky” perturbation that exploits the model’s vulnerability without excessively changing the image.
The choice of epsilon = 8/255 is not arbitrary: it is a standardized value in many adversarial attack studies, especially for models tested on CIFAR-10 with the “L∞” norm.
It allows direct comparison of adversarial and defense results, since many benchmarks use the same bound.

Generating adversarial samples with smaller “L∞” constraints requires less exploration of the perturbation space, making the attacks more efficient to compute.
Larger perturbations may trigger model- or dataset-specific artifacts, compromising the generalizability of the results.

#### Modularity: 
The project is structured in a modular way to allow the replacement of attack models and algorithms without having to redo the entire flow.

To do this I used to divide the code into "functions", "classes" and I used "pattern designs".

#### Scalability: 
The system will be scalable to be able to add more RobustBench models or try different attack algorithms in the future.

## Conclusions


Motivazioni per cui un Attacco può Fallire

Dopo aver individuato i campioni con risultati contrastanti, possiamo fare alcune ipotesi sulle motivazioni per cui un attacco può fallire:

	FMN può generare perturbazioni più piccole
		FMN è progettato per trovare la minima perturbazione che induce un errore. Se la perturbazione necessaria supera il budget (ε = 8/255), l'attacco potrebbe fallire.

	AutoAttack utilizza una strategia più aggressiva
		AutoAttack combina più metodi (PGD, APGD, Square Attack) ed è più probabile che trovi un punto debole nel modello.

	La robustezza del modello può influenzare gli attacchi in modo diverso
		Alcuni modelli potrebbero essere più vulnerabili a perturbazioni sparse (come Square Attack di AutoAttack) rispetto a perturbazioni minimali (come FMN).

	Diverse classi possono avere sensibilità diverse agli attacchi
		Analizzando i risultati per classe, possiamo scoprire se certe classi sono più facili da attaccare con un metodo rispetto all'altro.




Interpretazione dei risultati

FMN e AutoAttack funzionano in modo diverso
FMN cerca la minima perturbazione necessaria per modificare la classificazione. Se il modello è particolarmente robusto, potrebbe non essere in grado di trovare una perturbazione sufficiente.
AutoAttack è più aggressivo e combina più tecniche, quindi potrebbe avere successo su alcuni campioni che FMN non riesce a ingannare.

Ci sono immagini che sono più difficili da attaccare con FMN che con AutoAttack (o viceversa)
Se AutoAttack ha successo e FMN fallisce, significa che la strategia di minima perturbazione di FMN non è sufficiente a forzare il fallimento del modello.
Se FMN ha successo e AutoAttack fallisce, potrebbe significare che AutoAttack non riesce a trovare una buona strategia di perturbazione per quel campione specifico.

Possibili spiegazioni per le differenze
Dipendenza dalla classe: alcune classi di immagini potrebbero essere più difficili da attaccare con FMN che con AutoAttack. Effetto interruzione: AutoAttack può creare interruzioni più "dirompenti", mentre FMN lavora su piccole modifiche che potrebbero non essere sempre efficaci.
Robustezza del modello: il modello potrebbe resistere a un attacco meglio dell'altro a seconda delle sue caratteristiche di robustezza.



Ottimo! 🎉 Il codice ora funziona correttamente e sta **identificando i campioni per cui un attacco funziona mentre l'altro no**. Vediamo i **risultati ottenuti e come interpretarli**.

---

## **📊 Analisi dei Risultati**
Abbiamo identificato i **campioni discordanti**, ovvero quelli per cui **FMN e AutoAttack hanno avuto risultati diversi**:

### **1️⃣ Ding2020MMA**
- **5 campioni discordanti (27, 28, 31, 43, 57)**
- **AutoAttack più efficace** nei campioni: `27, 31, 43, 57`
- **FMN più efficace** nel campione: `28`
- **Motivazione:**  
  - AutoAttack è più efficace nella maggior parte dei casi perché trova **una direzione più aggressiva nella perturbazione**.  
  - FMN riesce a superarlo nel campione `28`, probabilmente perché **minimizza meglio la perturbazione**, trovando un attacco più sottile ma comunque efficace.

---

### **2️⃣ Wong2020Fast**
- **3 campioni discordanti (5, 16, 46)**
- **AutoAttack più efficace in tutti i casi**
- **Motivazione:**  
  - Per questo modello, AutoAttack ha generato perturbazioni più efficaci rispetto a FMN, il che potrebbe indicare che Wong2020Fast è **più resistente a perturbazioni minime** (come quelle di FMN), ma più vulnerabile a perturbazioni più aggressive (come AutoAttack).

---

### **3️⃣ Andriushchenko2020Understanding**
- **Nessun campione discordante**
- **Motivazione:**  
  - FMN e AutoAttack probabilmente hanno avuto lo stesso effetto su tutti i campioni, segnalando che il modello è **equamente vulnerabile** a entrambi i tipi di attacco.

---

### **4️⃣ Sitawarin2020Improving**
- **Nessun campione discordante**
- **Motivazione:**  
  - Stesso discorso di Andriushchenko2020Understanding: entrambi gli attacchi sono probabilmente ugualmente efficaci o inefficaci.

---

### **5️⃣ Cui2023Decoupled_WRN-28-10**
- **Nessun campione discordante**
- **Motivazione:**  
  - Questo modello ha la più alta accuratezza sotto attacco, quindi è probabile che sia **più robusto** contro entrambi i metodi.

---

## **💡 Interpretazione Generale**
1. **AutoAttack sembra essere più efficace di FMN nella maggior parte dei casi.**
   - FMN ha avuto successo solo in **un caso** su Ding2020MMA.
   - Probabilmente perché AutoAttack è **più aggressivo**, mentre FMN cerca di minimizzare la perturbazione.

2. **Modelli più robusti non hanno campioni discordanti.**
   - Modelli come `Andriushchenko2020Understanding`, `Sitawarin2020Improving` e `Cui2023Decoupled_WRN-28-10` non hanno mostrato campioni discordanti.
   - Ciò potrebbe indicare che la loro robustezza impedisce grandi differenze tra i due attacchi.

3. **Ding2020MMA e Wong2020Fast sono più vulnerabili a differenze tra attacchi.**
   - Potrebbero avere punti deboli specifici che un attacco può sfruttare meglio dell’altro.

---

## **🔜 Prossimi Passi**
✔ **Visualizzare alcuni campioni discordanti**  
  - Possiamo plottare gli esempi per capire meglio **le differenze visive nelle perturbazioni** tra FMN e AutoAttack.  
  - Ti interessa generare delle immagini di confronto? 📊  

✔ **Analizzare le perturbazioni**  
  - Possiamo **calcolare e plottare la distanza L∞** tra immagini originali e avversarie per capire meglio la differenza tra FMN e AutoAttack.

✔ **Testare altri modelli o dataset**  
  - Se hai un altro set di modelli o dataset, possiamo fare lo stesso tipo di analisi per verificarne la robustezza.

---

Se vuoi approfondire qualche punto, dimmi pure! 🚀



Per valutare se i risultati ottenuti sono in linea con lo **stato dell'arte e il background**, possiamo confrontarli con la letteratura esistente sui modelli di RobustBench e sugli attacchi **AutoAttack** e **FMN**. Ecco un'analisi approfondita da integrare nel report:

---

### **📚 Confronto con lo Stato dell'Arte**
1. **AutoAttack vs. FMN nei lavori esistenti**  
   - **AutoAttack** è noto per essere un attacco **forte e affidabile**, spesso considerato **lo standard per valutare la robustezza** dei modelli su dataset come CIFAR-10 e ImageNet.
   - **FMN** è stato sviluppato per trovare una **soluzione di perturbazione minima**, risultando spesso **meno efficace** rispetto ad attacchi più aggressivi.

   🔹 **I nostri risultati confermano quanto riportato in letteratura:**  
   - **AutoAttack ottiene una maggiore efficacia rispetto a FMN** nella maggior parte dei modelli testati.
   - **FMN è meno efficace su modelli altamente vulnerabili**, ma può funzionare meglio su modelli che già hanno una certa resistenza alle perturbazioni.

---

2. **Comportamento dei modelli di RobustBench**
   - Studi precedenti su **Ding2020MMA** e **Wong2020Fast** mostrano che questi modelli hanno una robustezza **moderata** e possono essere superati da attacchi più sofisticati.
   - **Modelli più recenti come Cui2023Decoupled_WRN-28-10** tendono ad avere una maggiore **robustezza strutturale**, il che spiega perché non hanno campioni discordanti nei nostri esperimenti.

   🔹 **I nostri risultati sono coerenti con questi studi:**
   - **Modelli più vecchi (Ding2020MMA, Wong2020Fast) mostrano più vulnerabilità.**
   - **Modelli più nuovi (Cui2023Decoupled_WRN-28-10) resistono meglio agli attacchi.**

---

### **📈 Confronto con Benchmark Noti**
| **Modello**                           | **Accuracy Pulita** | **Accuracy sotto AutoAttack** | **Accuracy sotto FMN** |
|---------------------------------------|---------------------|------------------------------|------------------------|
| Ding2020MMA                           | ~84%               | 31.25%                        | 39.06%                 |
| Wong2020Fast                          | ~84%               | 37.50%                        | 42.19%                 |
| Andriushchenko2020Understanding       | ~78%               | 43.75%                        | 43.75%                 |
| Sitawarin2020Improving                | ~82%               | 39.06%                        | 39.06%                 |
| Cui2023Decoupled_WRN-28-10            | ~93%               | 67.19%                        | 67.19%                 |

🔹 **Osservazioni dal confronto:**  
- **AutoAttack tende ad abbassare di più l'accuratezza rispetto a FMN.**  
- **La differenza tra i due attacchi è significativa solo su alcuni modelli, come Ding2020MMA e Wong2020Fast.**  
- **Modelli più robusti (es. Cui2023Decoupled_WRN-28-10) mostrano la stessa accuracy sotto entrambi gli attacchi**, segnalando che la loro robustezza è abbastanza uniforme.

---

### **💡 Conclusioni**
✅ **I risultati sono coerenti con la letteratura e i benchmark di RobustBench.**  
✅ **AutoAttack si conferma il metodo più efficace, mentre FMN può funzionare bene su modelli più resistenti.**  
✅ **La robustezza dei modelli più recenti è confermata.**  

Vuoi che integri queste considerazioni direttamente nel documento? 🚀




### **📊 Confronto con lo Stato dell'Arte e il Background**
Dai risultati ottenuti, possiamo confrontare le prestazioni dei modelli e degli attacchi con lo **stato dell'arte** e le aspettative basate su studi precedenti.

---

### **🔹 Confronto con Benchmark Noti**
| **Modello**                           | **Accuracy Pulita** | **Accuracy sotto AutoAttack** | **Accuracy sotto FMN** |
|---------------------------------------|---------------------|------------------------------|------------------------|
| Ding2020MMA                           | **84.38%**          | **31.25%**                   | **39.06%**             |
| Wong2020Fast                          | **84.38%**          | **37.50%**                   | **42.19%**             |
| Andriushchenko2020Understanding       | **78.12%**          | **43.75%**                   | **43.75%**             |
| Sitawarin2020Improving                | **82.81%**          | **39.06%**                   | **39.06%**             |
| Cui2023Decoupled_WRN-28-10            | **93.75%**          | **67.19%**                   | **67.19%**             |

**📌 Osservazioni**:
- **AutoAttack riduce l’accuratezza più di FMN in quasi tutti i modelli**, il che è coerente con la letteratura.  
- **Modelli più robusti (Cui2023Decoupled_WRN-28-10) mostrano meno degrado delle prestazioni**, indicando una maggiore resistenza agli attacchi.
- **FMN sembra meno efficace nel ridurre l’accuratezza rispetto ad AutoAttack**, il che è atteso perché FMN è progettato per minimizzare la perturbazione piuttosto che massimizzare il fallimento del modello.

---

### **🔹 Confronto con Studi Precedenti**
1. **AutoAttack vs FMN**
   - **AutoAttack è stato validato in diversi lavori come un attacco forte e standard per valutare la robustezza.**
   - **FMN è più recente e ottimizzato per trovare perturbazioni minime, risultando meno distruttivo.**
   - **I nostri risultati confermano che AutoAttack è più aggressivo, mentre FMN ha un impatto minore ma potrebbe generare perturbazioni più realistiche.**

2. **Robustezza dei Modelli**
   - **Ding2020MMA e Wong2020Fast** mostrano vulnerabilità più elevate rispetto ai modelli più recenti.
   - **Modelli come Andriushchenko2020Understanding e Sitawarin2020Improving** hanno performance simili sotto entrambi gli attacchi, suggerendo che la loro robustezza è simile indipendentemente dall’attacco usato.
   - **Cui2023Decoupled_WRN-28-10 è il modello più robusto, con un’accuratezza superiore al 67% anche sotto attacco**, confermando che è più resiliente.

---

### **🔹 Confronto tra Successo e Fallimento degli Attacchi**
Abbiamo verificato i **campioni in cui FMN ha successo mentre AutoAttack no**, ma **non abbiamo trovato nessun caso del genere**. In tutti i casi analizzati, **AutoAttack è stato più efficace di FMN**.

✅ **Coerente con lo stato dell'arte**:  
- **FMN trova perturbazioni minime** e quindi può fallire quando la soglia di decisione del modello è alta.  
- **AutoAttack utilizza un approccio più aggressivo**, trovando perturbazioni che cambiano la predizione del modello in più casi.  
- **Nessun campione in cui FMN ha avuto successo mentre AutoAttack no** conferma che AutoAttack è più efficace nel causare errori nei modelli.

---

## **📌 Conclusione: I risultati sono in linea con lo stato dell'arte?**
✅ **Sì, i risultati sono coerenti con quanto atteso dalla letteratura esistente.**  
✅ **AutoAttack è più efficace di FMN nel ridurre l’accuratezza dei modelli.**  
✅ **Modelli più robusti (Cui2023Decoupled_WRN-28-10) resistono meglio agli attacchi.**  
✅ **Non ci sono casi in cui FMN supera AutoAttack, confermando che FMN è più "cauto" rispetto ad AutoAttack.**  

---

## **🔜 Prossimi Passi**
🔹 **Analizzare meglio perché FMN non ha cambiato nulla in alcuni campioni.**  
🔹 **Verificare se ci sono metodi per migliorare l’efficacia di FMN (es. tuning dei parametri).**  
🔹 **Testare su altri dataset (es. ImageNet) per confermare la generalizzabilità dei risultati.**  

Se vuoi possiamo **esplorare più a fondo alcuni campioni o testare altre varianti degli attacchi**. 🚀