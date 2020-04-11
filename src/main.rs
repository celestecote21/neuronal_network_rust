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
    

    let mut network = Network::new(vec![784, 11, 16, 10], 3.0);

    for _ in 0..30{
        network.mini_batch(match train.next_chunk(){
            Ok(v) => v,
            Err(err) => panic!("error creating chunck {}", err),
        });
    }

    let test = match train.next_chunk(){
            Ok(v) => v,
            Err(err) => panic!("error creating chunck {}", err),
        };
    println!("reponce: {}, trouver{:?}", test[0].1, network.compute(test[0].0.to_vec()).unwrap().0.last().unwrap());

    
    println!("Hello, world!");
    
    // loop{}
}



