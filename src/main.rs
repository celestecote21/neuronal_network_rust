extern crate nalgebra;

mod network;
mod training_tuple;

// use image::{GenericImage, GenericImageView, ImageBuffer, RgbImage};
use network::Network;
use training_tuple::ImageTrain;



fn main() {
    
    let mut train = match ImageTrain::new("data.idx", "data_label.idx"){
        Ok(t) => t,
        Err(err) => panic!("error creation trainig data {}", err),
    };
    

    let mut network = Network::new(vec![784, 16, 16, 10]);

    network.mini_batch(match train.next_chunk(){
        Ok(v) => v,
        Err(err) => panic!("error creating chunck {}", err),
    });
    
    println!("Hello, world!");
    
    // loop{}
}



