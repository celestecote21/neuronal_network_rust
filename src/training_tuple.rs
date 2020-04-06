


// use image::{GenericImage, GenericImageView, ImageBuffer, RgbImage};
use std::fs::File;
use std::io::Read;
use rand::prelude::*;


pub struct ImageTrain{
    raw_data: Vec<u8>,
    raw_label: Vec<u8>,
    nb_dim: u8,
    size_dim: Vec<u32>,
    nb_label: usize,
    vec_tuple: Vec<(Vec<f64>, u8)>,
    vec_used: Vec<bool>,
    chunk_size: usize,
    current_chunk: usize,
}


impl ImageTrain{
    pub fn new(file_name: &str, file_name_label: &str) -> Result<ImageTrain, String>{
        let mut temp = ImageTrain{
            raw_data: Vec::new(),
            raw_label: Vec::new(),
            nb_dim: 0,
            size_dim: Vec::new(),
            nb_label: 0,
            vec_tuple: Vec::new(),
            vec_used: Vec::new(),
            chunk_size: 10,
            current_chunk: 0,
        };
        match temp.load_file(file_name, file_name_label){
            Ok(t) => t,
            Err(err) => return Err(err),
        };
        Ok(temp)
    }

    pub fn load_file(&mut self, file_name: &str, file_name_label: &str) -> Result<u8, String>{

        let mut tab_precalcul = Vec::new();
        let coef = 2.0 / 255.0;
        for i in 0..256{
            tab_precalcul.push(i as f64 * coef - 1.0);
        }

        let mut f = match File::open(file_name){
            Ok(file) => file,
            Err(_) => return Err(String::from("can't open the file img")),
        };
        let mut file_lab  = match File::open(file_name_label){
            Ok(file) => file,
            Err(_) => return Err(String::from("can't open  the file label")),
        };

        match f.read_to_end(&mut self.raw_data){ // lecture du ficher pour que tout les valeurs soit dans un vector
            Ok(_) => (),
            Err(erro) => return Err(format!("unable to read the document {:?}", erro)),
        };

        match file_lab.read_to_end(&mut self.raw_label){ // lecture du ficher pour que tout les valeurs soit dans un vector
            Ok(_) => (),
            Err(erro) => return Err(format!("unable to read the document {:?}", erro)),
        };

        //________________________________________Lecture header image____________________//
        let mut index = 0;
        index += 2; // le fichier commencer par 2 octet = 0
        if self.raw_data[index] != 0x08 && self.raw_data[index + 1] != 0x03{
            panic!("file not good");
        }
        // le prochain byte c'est le nombre de dimension dans la matrix
        index += 1;
        self.nb_dim = self.raw_data[index];

        // recuperation des tailles des dimensions
        index += 1;
        for _ in 0..self.nb_dim{
            let mut nb_el: u32 = 0;
            for _ in 0..4{
                nb_el <<= 8;
                nb_el = nb_el | self.raw_data[index] as u32;
                index += 1;
            }
            // println!("{}", nb_el);
            self.size_dim.push(nb_el);
        }

        let start_addr = index; // la derniere addresse visiter, la premiere image commence ici
        //_____________________________________Lecture header label_______________________//
        
        index = 2; // le fichier commencer par 2 octet = 0
        if self.raw_label[index] != 0x08 && self.raw_label[index + 1] != 0x01{
            return Err("header file of label is not good".to_string());
        }
        // le prochain byte c'est le nombre de dimension dans la matrix
        index += 2;
        let mut nb_el: u32 = 0;
        for _ in 0..4{
            nb_el <<= 8;
            nb_el = nb_el | self.raw_label[index] as u32;
            index += 1;
        }
        self.nb_label = nb_el as usize;
        let start_addr_label = index;
        
        //_____________________________________CREATION TUPLES__________________________//
        
        println!("lecture et conversion des fichier");


        if self.nb_label != self.size_dim[0] as usize{
            return Err("the number of label and of image is not the same".to_string());
        }
        let taille_img = self.size_dim[1] as usize * self.size_dim[2] as usize;

        for i in 0..self.nb_label as usize{
            let v_temp = self.raw_data[(i*taille_img + start_addr)..((i+1)*taille_img + start_addr)].to_vec();
            let lab_temp = self.raw_label[i+start_addr_label];
           // convertion des donneer en un vector d'intervall [-1;1]
           // donc passsage en f64
           let mut vec_float =  Vec::new();
           for i in v_temp.iter(){
                vec_float.push(tab_precalcul[*i as usize]);
           }
            self.vec_tuple.push((vec_float, lab_temp));
        }

        // melange des tulpes
        let mut rng = rand::prelude::thread_rng();
        self.vec_tuple.shuffle(&mut rng);
        println!("conversion et melange des fichier terminer");
        Ok(1)

    }

    pub fn next_chunk(&mut self, chunk_size: usize) -> Result<Vec<(Vec<f64>, u8)>, String>{

        if self.current_chunk + self.chunk_size > self.vec_tuple.len(){
            return Err("end of file".to_string());
        }
        // calcul des indexs dans le vector
        
        let new_vec = (&self.vec_tuple[self.current_chunk..self.current_chunk + self.chunk_size]).to_vec(); // le nouveau vector avec chunk_size de training
        self.current_chunk += self.chunk_size;
        Ok(new_vec)
     } 
}
