extern crate nalgebra;

mod network;
mod training_tuple;

// use image::{GenericImage, GenericImageView, ImageBuffer, RgbImage};
use std::fs::File;
use std::io::Read;
use network::Network;
use training_tuple::ImageTrain;
use rand::prelude::*;



fn main() {
    
    let mut train = match ImageTrain::new("data.idx", "data_label.idx"){
        Ok(t) => t,
        Err(err) => panic!("error creation trainig data {}", err),
    };
    

    let mut network = Network::new(vec![784, 16, 16, 10]);
    
    println!("Hello, world!");
    
    // loop{}
}


fn view_img(img: &Vec<u8>){
    let mut index = 0;
    for _ in 0..28{
        for _ in 0..28{
            if img[index] >= 125 {
                print!("$");
            }else{
                print!("O");
            }
            index += 1;
        }
        print!("\n");
    }
}
