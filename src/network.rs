extern crate rand;

mod linear_math;

use rand::prelude::*;
use rand_distr::StandardNormal;
use linear_math as lm; 

pub struct Network{
    nb_layer: usize,
    layers_size: Vec<u32>,
    bias: Vec<Vec<f32>>,
    weight: Vec<Vec<Vec<f32>>>,
}


impl Network{

    pub fn new(size: Vec<u32>) -> Network{
        let mut bias_temp = Vec::new();
        let mut weight_temp = Vec::new();
        // creation bias vector
        for i in 1..size.len(){
           let mut v_b = Vec::new();
           for _ in 0..size[i]{
               v_b.push(thread_rng().sample(StandardNormal));
           }
           bias_temp.push(v_b);
        }

        // creation weight vector
        
        for i in 0..size.len() - 1{
            let mut v_wd = Vec::new();
            for _dn in 0..size[i]{
                let mut v_wdn = Vec::new();
                for _fn in 0..size[i+1]{
                    v_wdn.push(thread_rng().sample(StandardNormal));
                }
                v_wd.push(v_wdn);
            }
            weight_temp.push(v_wd);
        }

        Network{
            nb_layer: size.len(),
            layers_size: size.to_vec(),
            bias: bias_temp,
            weight: weight_temp,
        }
    }

    pub fn mini_batch(&mut self, mini_batch: Vec<(Vec<f32>, u8)>){
        for i in mini_batch.iter(){
            // let mut delta_out = Vec::new();

            println!("{:?}", *i);
        }
    }

    pub fn compute(&self, mut input: Vec<f32>) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), i8>{
        let mut zl_x = Vec::new();
        let mut al_x = Vec::new();

        for current_dim in 0..self.nb_layer - 1{
            if input.len() != self.layers_size[current_dim] as usize{
                return Err(-1);
            }
            let mut v_inter = Vec::new();
            for current_neur in 0..self.layers_size[current_dim + 1] as usize{
                let mut somme = 0.0;
                for i in 0..input.len(){
                    somme += input[i] * self.weight[current_dim][i][current_neur]; 
                    // let test:f64 = self.weight[current_dim][i][current_neur]; 
                }
                v_inter.push(somme + self.bias[current_dim][current_neur])
            } // repeter pour chaque dimension normalement en remplacent just l'input c'est good
            zl_x.push(v_inter.clone());

            input = Network::sigmoid(&v_inter);
            al_x.push(input.clone());
        }
        Ok((al_x, zl_x))
    }

    fn sigmoid(v: &Vec<f32>) -> Vec<f32>{
        // println!("before {:?}", *v);
        let mut result = Vec::new();
        for i in v.iter(){
            result.push(1.0/(1.0 + (-i).exp()))
        }
        // println!("after {:?}", result);
        result
    }
}
