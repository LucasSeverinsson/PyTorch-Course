#### NOTES

### Week 1

## Exercise A 
# A1. shapes and ranks of torch.tensor(3), torch.arange(7), torch.zeros(2,1,4)
torch.tensor(3): scalar with shape (), rank 0
torch.arange(7): 1 dimensional, rank 1, with shape (6,)
torch.zeros(2,1,4): rank 4 shape (2, 1, 4)

# A2. how many elements and what does axis 1 refer to?
shape(5,8,3): numel = 120, axis refer to how many rows for each stacked matrix

# A3. matrix = torch.tensor[[1,2,3],[4,5,6],[7,8,9]] Extract its second row and third column using indexing/slicing. 
second_row = matrix[1]
third_column = matrix[:,2]

# A4.

## 1.1 What is a tensor

A multidimensional array

# Why are tensors central to deep learning?
