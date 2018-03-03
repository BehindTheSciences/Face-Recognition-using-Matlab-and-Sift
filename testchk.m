function [ N, M ] = testchk( image )
%This function compare a test image with the training data.
%   Uses Classifier based on BBF search algorithm proposed by Lowes.
% Input is the test image number
% Outputs: M is the matching person number
%          N is the training image number with which test image has most
%          matches with.

no_img = 170; %number of training images

   distRatio = 0.6;  %threshold which determine which matches to keep.

   imag = sprintf('testset/image_%03d.pgm',image);

  %This part detect the face region and discard other parts such as
  %background.
  img = ObjectDetection(imag,'HaarCascades/haarcascade_frontalface_alt.mat');

  imwrite(img,'caa.pgm','pgm'); %save the cropped test face image to a temporary file.
  
  


[im, des, locs] = sift('caa.pgm'); %Extract the SIFT features of the test image.
out = load(sprintf('Training1/data1.mat')); %loads the training images features.


for count=1:1:no_img

    
    desc = out.output{count,1}.descriptors; %load the descriptor of the training image
    dest = desc';                         % Precompute matrix transpose
    
for i = 1 : size(des,1)
   dotprods = des(i,:) * dest;        % Computes vector of dot products
   [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results

   % Check if nearest neighbor has angle less than distRatio times 2nd.
   if (vals(1) < distRatio * vals(2))
      match(i) = indx(1);
   else
      match(i) = 0;
   end
end
    
Kpoint(1,count) = sum(match > 0); %compute the total number of matches found.
fprintf('Found %d matches with training image %d.\n', Kpoint(1,count),count);  
    
    
    
end



[B,IX] = sort(Kpoint,'descend'); %Compute the highest number of matches
fprintf('\n\n');




for count=1:1:no_img
fprintf('Found %d matches with training image %d.\n', B(count),IX(count));
end




    %This gives out the person which was identify as the test image subject
    out = load('training1/data1.mat');
    N = IX(1);                              %training image no
    M = out.output{N,1}.person;             %person no
    fprintf('The image correspond to person %d\n', M);

figure;
imshow(img);
title('Subject Image')
figure;
imshow(out.output{N,1}.image);
title(sprintf('The image correspond to person %d', M))

end
