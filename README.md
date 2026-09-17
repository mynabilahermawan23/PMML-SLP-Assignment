# Tugas PMML - Single Layer Perceptron

Repository ini berisi kode Python yang digunakan untuk mengerjakan tugas PMML mengenai **Single Layer Perceptron (SLP)**.

## Pembahasan

Kode mencakup:
- proses training dan validation,
- perhitungan accuracy setiap epoch,
- perhitungan loss setiap epoch,
- serta pembuatan grafik accuracy dan loss.

Model SLP dibangun mengikuti rumus yang sama seperti perhitungan manual di Google Sheets. Hasil dari Python kemudian dibandingkan dengan hasil pada Google Sheets. Nilai yang diperoleh digunakan untuk membuat grafik training dan validation selama 5 epoch.

## Tools

- Python
- NumPy
- Matplotlib
- SciPy

## Cara Menjalankan

1. Install library yang dibutuhkan:
   ```
   pip install numpy matplotlib scipy
   ```
2. Pastikan file dataset (`iris_slp_data.csv`) berada satu folder dengan `slp_iris.py`.
3. Jalankan script:
   ```
   python slp_iris.py
   ```
4. Grafik accuracy dan loss akan otomatis tersimpan di folder `result/`.

## Author

Nama: Nabila Hermawan
Program Studi: Magister Kecerdasan Artifisial

## Daftar Pustaka

Afiahayati. *Linear Classifier - Single Layer Perceptron*. Materi Kuliah, Departemen Ilmu Komputer dan Elektronika, Universitas Gadjah Mada.

Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. Annals of Eugenics, 7(2), 179-188. 

UCI Machine Learning. Iris Species Dataset. Kaggle. https://www.kaggle.com/datasets/uciml/iris 

Scikit-learn Developers. *Perceptron - Scikit-learn Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
