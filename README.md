# AP_test
This is the GitHub Repo for Allocation Probability Test application on TS-PostDiff


              #each time batch size is "crossed", m == 0, rbeta parameters are
        #updated to include most current success/failure counts, post prob
      #when m != 0, we stick with info from previous batch, older post prob
    #why rbeta? most suitable for binary reward system such as TSPDD


              #same TS updating process done here again but on a more local level, 
        #if the effect size is better than the threshold, it shows we are 
      #willing to go back to exploiting again, when effect size is smaller, 
    #we are open to doing more exploration with UR