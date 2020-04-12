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
    

    let mut network = Network::new(vec![784, 16, 16, 10], 0.5);

    for _ in 0..30{
        for _ in 0..5000{
            network.mini_batch(match train.next_chunk(){
                Ok(v) => v,
                Err(err) => panic!("error creating chunck {}", err),
            });
        }

        let mut nb_just = 0;
        let nb_img = 1000;
        for t in 0..nb_img{
            let test = match train.next_chunk(){
                    Ok(v) => v,
                    Err(err) => panic!("error creating chunck {}", err),
                };
            nb_just += network.test(test)
        }
        println!("{} / {}", nb_just, 10*nb_img);
        train.reset();
    }
    
    println!("Hello, world!");
    
    // loop{}
}



