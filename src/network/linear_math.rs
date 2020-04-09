
use std::ops::{Sub, Mul};


pub fn reponse_to_vec(responce: usize, size: usize) -> Vec<f32>{
    let mut v = Vec::new();
    for i in 0..size{
        if i == responce{
            v.push(1.0);
        }else{
            v.push(0.0);
        }
    } 
    v
}

pub fn soustraction<T>(a: Vec<T>, b: Vec<T>) -> Result<Vec<T>, String>
    where T: Copy + Sub<Output = T> 
{
    if a.len() != b.len(){
        return Err("the Vec don't have the same size".to_string());
    }
    let mut v = Vec::new();
    for i in 0..a.len(){
        v.push(a[i] - b[i]);
    }
    Ok(v)
}


pub fn scalar_mult<T>(scalar: T, v: Vec<T>) -> Vec<T>
    where T: Copy + Mul<Output = T>
{
    let result: Vec<_> = v.iter().map(|x| *x * scalar).collect();
    result
}

pub fn hadamard<T>(a: Vec<T>, b: Vec<T>) -> Result<Vec<T>, String>
    where T: Copy + Mul<Output = T>
{
    if a.len() != b.len(){
        return Err("the Vec don't have the same size".to_string());
    }
    // TODO: essayer d'avoir un algo assez rapide
    Ok(a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect())

}

// pub fn norm<T>(a: Vec<T>) -> Vec<T>{
    // 
// }



#[cfg(test)]
mod tests_vec{

    use crate::network::linear_math::*;

    #[test]
    fn test_sub(){
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 2, 2];
        let c = vec![0, 0, 1, 2];
        assert_eq!(crate::network::linear_math::soustraction(a, b).unwrap(), c);
    }
    #[test]
    fn test_responce(){
        let y = 4;
        let size = 10;
        assert_eq!(crate::network::linear_math::reponse_to_vec(y, size), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_elementwise(){
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 2, 2];
        let c = vec![1, 4, 6, 8];

        assert_eq!(hadamard(a, b).unwrap(), c);
    }
}

