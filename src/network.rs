extern crate rand;

mod linear_math;

use rand::prelude::*;
use rand_distr::StandardNormal;
use linear_math as lm; 

pub struct Network{
    nb_layer: usize,
    mu: f32,
    layers_size: Vec<u32>,
    bias: Vec<Vec<f32>>,
    weight: Vec<Vec<Vec<f32>>>,
}


impl Network{

    pub fn new(size: Vec<u32>, mu: f32) -> Network{
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
            let mut w_out = Vec::new();
            for _dn in 0..size[i + 1]{
                let mut w_in = Vec::new();
                for _fn in 0..size[i]{
                    w_in.push(thread_rng().sample(StandardNormal));
                }
                w_out.push(w_in);
            }
            weight_temp.push(w_out);
        }

        Network{
            nb_layer: size.len(),
            layers_size: size.to_vec(),
            bias: bias_temp,
            weight: weight_temp,
            mu,
        }
    }

    pub fn mini_batch(&mut self, mini_batch: Vec<(Vec<f32>, u8)>){
        let mut delta_total = Vec::new();
        let mut a_total = Vec::new();
        for i in mini_batch.iter(){
            let mut layer = self.weight.len() - 1;
            let mut delta = Vec::new();
            let y = lm::reponse_to_vec(i.1 as usize, 10);
            let result_compute = self.compute(i.0.to_vec()).unwrap();
            let a_last = result_compute.0.last().unwrap().to_vec();
            let z_last = result_compute.1.last().unwrap().to_vec();
            let mut z_al = result_compute.1.to_vec();
            z_al.pop();

            let delta_out = lm::scalar_mult(2.0, lm::soustraction(a_last, y).unwrap());
            let mut delta_out = match lm::hadamard(delta_out, Network::sigmoid_deri_v(z_last)){
                Ok(v) => v,
                Err(err) => panic!("error on delta out {}", err),
            };

            delta.insert(0, delta_out); // on met le delta de la sortie dernier (c'est le premier donc voila mais par principe)
            if layer <= 0{
                panic!("implemeter que 2 layer");
            }
            layer += 1;
            for current_dim in (2..layer + 1).rev(){
                // println!("{:?}", current_dim);
                let mut delta_current_dim = Vec::new();
                for neur_L in 0..(self.layers_size[current_dim - 1]) as usize{
                    let mut delta_n = 0.0;
                    for neur_L1 in 0..(self.layers_size[current_dim]) as usize{
                        delta_n += delta[0][neur_L1] * self.weight[current_dim - 1][neur_L1][neur_L];
                    }
                    delta_current_dim.push(delta_n);
                }
                // println!("{}", delta_current_dim.len());
                // println!("{:?}", z_al.last().unwrap());
                delta.insert(0, lm::hadamard(Network::sigmoid_deri_v(z_al.pop().unwrap()),delta_current_dim).unwrap());
            }
            // println!("{}", delta.len());
            delta_total.push(delta);
            a_total.push(result_compute.0);
        }
        // println!("{:?}", delta_total);
        self.change_b_w(delta_total, a_total);
    }


    fn change_b_w(&mut self, delta_total: Vec<Vec<Vec<f32>>>, a_total: Vec<Vec<Vec<f32>>>) -> Result<i8, String>{
        let im_layer = delta_total[0].len();
        // println!("{:?}", im_layer);
        let mut n_bias = Vec::new();
        for i in 0..im_layer{ // pour chaque layer
            // println!("i: {}", i);
            let im_neurone = delta_total[0][i].len(); // nombre de neurone dans le layer
            // println!("im_neurone: {}", im_neurone);
            let mut bias_1_layer = Vec::new(); 
            for j in 0..im_neurone{ // pour chaque neurone dans un layer
                let mut somme = 0.0;
                for k in 0..(delta_total.len()){ // pour chaque image
                    somme += delta_total[k][i][j]; // on ajoute donc le delta d'un meme neurone a travers les 10 images
                }
                let moyenne  = (self.mu * somme) / delta_total.len() as f32; // on veut la moyenne des image , et on multiplie par mu
                bias_1_layer.push(self.bias[i][j] - moyenne); // on met le resultat dans le vector, vector de la moyenne des deltas de tout les neurones d'un layer 
            }
            n_bias.push(bias_1_layer); // vector de tous les layers
        }
        // println!("{:?} n_bias\n", n_bias);
        self.bias = n_bias;

        let mut new_weight = Vec::new();
        // println!("{}", self.weight.len());
        let im_layer = delta_total[0].len();
        for dim in 0..im_layer{ // pour chaque layer
            let mut out = Vec::new();
            for i in 0..delta_total[0][dim].len(){ // pour chaque erreur dans neurone dans L OUT
                let mut in_v = Vec::new();
                for j in 0..a_total[0][dim].len(){ // pour chaque activation dans les neurones dans L-1 IN
                    let mut somme = 0.0;
                    for k in 0..delta_total.len(){ // pour tout les images on prend la somme
                       somme += delta_total[k][dim][i] * a_total[k][dim][j];
                    }
                    in_v.push(self.weight[dim][i][j] - (self.mu*somme)/delta_total.len() as f32);
                }
                out.push(in_v);
            }
            // println!("{:?}", out.len());
            new_weight.push(out);
        }
        // println!("{:?}", new_weight);
        self.weight = new_weight;
        Ok(1)        
    }
//                                                         al                zl
    pub fn compute(&self, mut input: Vec<f32>) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), i8>{
        let mut zl_x = Vec::new();
        let mut al_x = Vec::new();

        al_x.push(input.clone());

        for current_dim in 0..self.nb_layer - 1{
            if input.len() != self.layers_size[current_dim] as usize{
                return Err(-1);
            }
            let mut v_inter = Vec::new();
            for current_neur in 0..self.layers_size[current_dim + 1] as usize{
                let mut somme = 0.0;
                for i in 0..input.len(){
                    somme += input[i] * self.weight[current_dim][current_neur][i]; 
                    // let test:f64 = self.weight[current_dim][i][current_neur]; 
                }
                v_inter.push(somme + self.bias[current_dim][current_neur])
            } // repeter pour chaque dimension normalement en remplacent just l'input c'est good
            zl_x.push(v_inter.clone());

            input = Network::sigmoid_v(&v_inter);
            al_x.push(input.clone());
        }
        // println!("{:?} \n", al_x);
        Ok((al_x, zl_x))
    }

    pub fn test(&self, chunk: Vec<(Vec<f32>, u8)>) -> u32{
        let mut score = 0;
        for img in chunk.iter(){
            let mut input = img.0.to_vec();
            for current_dim in 0..self.nb_layer - 1{
                if input.len() != self.layers_size[current_dim] as usize{
                    panic!("panic in test");
                }
                let mut v_inter = Vec::new();
                for current_neur in 0..self.layers_size[current_dim + 1] as usize{
                    let mut somme = 0.0;
                    for i in 0..input.len(){
                        somme += input[i] * self.weight[current_dim][current_neur][i]; 
                        // let test:f64 = self.weight[current_dim][i][current_neur]; 
                    }
                    v_inter.push(somme + self.bias[current_dim][current_neur])
                } // repeter pour chaque dimension normalement en remplacent just l'input c'est good

                input = Network::sigmoid_v(&v_inter);
            }
            // println!("n: {} v: {:?}", img.1, input);
            let reponce = lm::vec_to_nb(input);
            if reponce == img.1{
                score += 1;
            }
        }
        score
    }

    fn sigmoid(x: f32) -> f32{
        1.0/(1.0 + (-x).exp())
    }

    fn sigmoid_v(v: &Vec<f32>) -> Vec<f32>{
        v.iter().map(|x| Network::sigmoid(*x)).collect()
    }

    fn sigmoid_deri_v(v: Vec<f32>) -> Vec<f32>{
        v.iter().map(|x| Network::sigmoid(*x) * (1.0 - Network::sigmoid(*x))).collect()
    }
}
