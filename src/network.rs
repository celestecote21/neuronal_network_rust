extern crate rand;

use rand::prelude::*;
use rand_distr::StandardNormal;


pub struct Network{
    nb_layer: usize,
    layers_size: Vec<u32>,
    bias: Vec<Vec<f64>>,
    weight: Vec<Vec<Vec<f64>>>,
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

    pub fn compute(&self, mut input: Vec<f64>) -> Result<Vec<f64>, i8>{
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
            input = Network::sigmoid(&v_inter);
        }
        Ok(input)
    }

    fn sigmoid(v: &Vec<f64>) -> Vec<f64>{
        // println!("before {:?}", *v);
        let mut result = Vec::new();
        for i in v.iter(){
            result.push(1.0/(1.0 + (-i).exp()))
        }
        // println!("after {:?}", result);
        result
    }
}
