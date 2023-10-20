# AI Image Denoiser
<p><i>Welcome to the AI Image Denoiser project, an innovative deep learning solution for image noise reduction and automatic image remastering. This project leverages a custom deep neural network (DNN) model, built using the powerful Dlib library for image processing and AI.</i></p>
<p><i>Our model is capable of removing various types of noise and grain from images and can even be used for automatic video stream remastering, thanks to its integration with the latest FFMPEG-based functionalities in the Dlib library.</i></p>

<h2>Description</h2>
<p>The AI Image Denoiser model is inspired by the "Pix2Pix" principle and employs a neural architecture that combines a U-Net structure and a ResNet-style neural typology. It excels in processing artificially noised images, supporting three noise types: flow, occlusion, and Gaussian noise. The model processes the input image and produces an enhanced, noise-reduced output. While it currently operates on grayscale images, adapting it for color images is a straightforward extension.</p>

<h2>Usage</h2>
<p>This AI Image Denoiser is an open-source project released under the GPL license, making it freely available for the community to use and contribute to. It comes with a robust training and model generation process, offering two key parameters for immediate usage and the possibility of fine-tuning the provided model.</p>
<p>To utilize the AI Image Denoiser tool, you can use the following parameters:
<ol>
  <li><code>--train &#91;directory&#93;</code>: this parameter enables you to initiate (or continue) the training process for the model;</li>  
  <li><code>--test &#91;directory or image path&#93;</code>: this parameter allows for immediate model usage. You can provide either a directory of images or a single image for denoising.</li>  
</ol>
</p>

<h2>Official Dlib Models Repository</h2>
<p>Our model has also been published directly on the official <a href="https://github.com/davisking/dlib-models">Dlib models repository</a>. However, this current GitHub page will always provide you with the very latest version and updates of the AI Image Denoiser.</p>
<p><b>Community Contributions:</b> if you find that fine-tuning the model yields superior results, don't hesitate to request the insertion of the new model via the creation of a Pull Request (PR) on this page. We value your input and are eager to incorporate improvements suggested by the community.</p>
